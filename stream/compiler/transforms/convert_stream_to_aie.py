from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from snaxc.dialects.snax import LayoutCast
from snaxc.dialects.tsl import TiledStridedLayoutAttr
from snaxc.ir.tsl import Stride, TiledStridedLayout
from xdsl.context import Context
from xdsl.dialects import scf
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    MemRefType,
    ModuleOp,
    ShapedType,
)
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa
from xdsl_aie.dialects.aie import (
    BDDimLayout,
    BDDimLayoutArray,
    BDDimLayoutArrayAttr,
    Block,
    CoreOp,
    DeviceOp,
    EndOp,
    ObjectFifoAcquireOp,
    ObjectFifoLinkOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    ObjectFIFOSubview,
    ObjectFIFOSubviewAccessOp,
    SymbolTable,
    TileOp,
)

from stream.compiler.context.aie_context import AIEContext
from stream.compiler.dialects.stream import (
    ComputationNodeOp,
)
from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_aie_kernels import ConvertAIEKernels


def canonicalize_transformation(sizes: Sequence[int], strides: Sequence[int]) -> tuple[list[int], list[int]]:
    """
    Examples:

        Size 1 can be omitted:
        [1, 1], [1, 1] -> [], []
        [4, 1], [1, 1] -> [4], [1]
        [1, 4], [4, 1] -> [4], [1]

        Squash redundancy:
        [4, 4], [4, 1] -> [16], [1]

    """

    resulting_strides: list[int] = []
    resulting_sizes: list[int] = []

    for size, stride in zip(reversed(sizes), reversed(strides), strict=False):
        assert size != 0
        if size == 1:
            continue
        if not resulting_sizes:
            resulting_sizes.insert(0, size)
            resulting_strides.insert(0, stride)
            continue
        # check for squash
        if stride == resulting_sizes[0] * resulting_strides[0]:
            resulting_sizes[0] *= size
        else:
            resulting_sizes.insert(0, size)
            resulting_strides.insert(0, stride)

    return resulting_sizes, resulting_strides


@dataclass
class SetKernelLayouts(RewritePattern):
    kernels: dict[str, AIEKernel]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        aie_kernel = self.kernels.get(op.kernel.data)
        assert aie_kernel is not None
        layouts = aie_kernel.operand_layouts()
        if not layouts:
            return
        shaped_operands = [operand for operand in op.operands if isinstance(operand.type, ShapedType)]
        for layout, operand in zip(layouts, shaped_operands, strict=True):
            assert isa(old_type := operand.type, MemRefType[FixedBitwidthType])
            layout_attr = TiledStridedLayoutAttr(layout)
            if old_type.layout == layout_attr:
                continue
            new_type = MemRefType(
                old_type.element_type,
                old_type.shape,
                layout_attr,
                old_type.memory_space,
            )
            new_operand = LayoutCast(operand, new_type)
            rewriter.insert_op(new_operand, InsertPoint.before(op))
            operand.replace_by_if(new_operand.results[0], lambda use: use.operation is op)


@dataclass
class HoistLayoutCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter) -> None:
        assert isinstance(op.source, OpResult)
        if isinstance(op.source.op, ObjectFIFOSubviewAccessOp):
            # good, this is what we want
            return
        elif isinstance(switch := op.source.op, IndexSwitchOp):
            # push up layout cast
            for case in (switch.default_region, *switch.case_regions):
                yield_op = case.block.last_op
                assert isinstance(yield_op, YieldOp)
                yielded = yield_op.arguments[0]
                assert isa(op.dest.type, MemRefType[FixedBitwidthType])
                new_cast = LayoutCast(yielded, op.dest.type)
                yield_op.operands[0] = new_cast.dest
                assert isinstance(yielded.owner, Operation)
                rewriter.insert_op(new_cast, InsertPoint.after(yielded.owner))
            switch.results[0].type = op.dest.type
            op.dest.replace_by(switch.output[0])
            rewriter.erase_op(op)


@dataclass
class SquashLayoutCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter) -> None:
        layout_casts = [use.operation for use in op.source.uses if isinstance(use.operation, LayoutCast)]
        # all dest types must be equal
        assert all(op.dest.type == cast.dest.type for cast in layout_casts)
        # keep only this one
        for cast_to_remove in filter(lambda x: x is not op, layout_casts):
            cast_to_remove.dest.replace_by(op.dest)
            rewriter.erase_op(cast_to_remove)


def get_transform(source: TiledStridedLayout, dest: TiledStridedLayout) -> tuple[list[int], list[int]]:
    """
    Returns sizes, strides
    """

    # list of dim, depth
    keys: list[tuple[int, int]] = []

    for dim in range(source.dimension()):
        for depth in range(source.tstrides[dim].depth()):
            keys.append((dim, depth))

    strides: list[dict[str, Stride]] = []

    for key in keys:
        strides.append(
            {
                "stride_src": source.get_stride(*key),
                "stride_dest": dest.get_stride(*key),
            }
        )

    strides.sort(key=lambda x: x["stride_dest"].step or 0, reverse=True)

    sizes_src, strides_src = zip(*[(x["stride_src"].bound, x["stride_src"].step) for x in strides], strict=False)
    sizes_dest, strides_dest = zip(
        *[(x["stride_dest"].bound, x["stride_dest"].step) for x in strides],
        strict=False,
    )

    # canonicalize
    sizes_src, strides_src = canonicalize_transformation(sizes_src, strides_src)
    sizes_dest, strides_dest = canonicalize_transformation(sizes_dest, strides_dest)

    # we only consider transformations at the source for now, so no transform should be happening at dest
    if len(sizes_dest) != 1:
        raise RuntimeError("did not expect dest transformation")

    return (sizes_src, strides_src)


@dataclass
class RealizeLayoutCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        device_op = op
        while not isinstance((device_op := device_op.parent_op()), DeviceOp):
            assert device_op is not None
        # gather some variables
        assert isinstance(op.source, OpResult)
        assert isinstance(subview_access := op.source.op, ObjectFIFOSubviewAccessOp)
        # assert isinstance(subview_access := op.source.op, ObjectFIFOSubviewAccessOp)
        assert isinstance(subview_access.subview, OpResult)
        assert isinstance(of_acquire := subview_access.subview.op, ObjectFifoAcquireOp)
        of_name = of_acquire.objFifo_name.root_reference.data

        if op.dest.type == op.source.type:
            op.dest.replace_by(op.source)
            rewriter.erase_matched_op()
            return

        def all_acquires(of: str) -> Iterable[ObjectFifoAcquireOp]:
            for walk_op in device_op.walk():
                if isinstance(walk_op, ObjectFifoAcquireOp) and walk_op.objFifo_name.root_reference.data == of:
                    yield walk_op

        # get all acquires and releases
        consumers: list[ObjectFifoAcquireOp] = []
        producers: list[ObjectFifoAcquireOp] = []
        for acquire in all_acquires(of_name):
            match ObjectFifoPortEnum.from_int(acquire.port.value.data):
                case ObjectFifoPortEnum.Consume:
                    consumers.append(acquire)
                case ObjectFifoPortEnum.Produce:
                    producers.append(acquire)

        def gather_layout(
            acquires: Sequence[ObjectFifoAcquireOp],
        ) -> MemRefType[FixedBitwidthType] | None:
            result = []
            for acquire in acquires:
                for subview in acquire.result.uses:
                    if isinstance(subview.operation, ObjectFIFOSubviewAccessOp):
                        for cast in subview.operation.output.uses:
                            if isinstance(cast.operation, LayoutCast):
                                dest_type = cast.operation.dest.type
                                assert isa(dest_type, MemRefType[FixedBitwidthType])
                                result.append(dest_type)
            if len(result) == 0:
                return None
            else:
                assert all([x == result[0] for x in result])
                return result[0]

        consumer_type = gather_layout(consumers)
        producer_type = gather_layout(producers)

        # create row-major layouts for those without explicit casts:

        if consumer_type is None:
            assert producer_type is not None
            assert isinstance(producer_type.layout, TiledStridedLayoutAttr)
            producer_layout = producer_type.layout.data
            strides = [1]
            for size in reversed(producer_type.shape.data[1:]):
                strides = [size.data * strides[0]] + strides
            assert isinstance(producer_type.layout, TiledStridedLayoutAttr)
            tile_bounds = producer_type.layout.data.tile_bounds()
            consumer_layout = TiledStridedLayout.from_strides(strides, tile_bounds)  # pyright: ignore
        elif producer_type is None:
            assert consumer_type is not None
            assert isinstance(consumer_type.layout, TiledStridedLayoutAttr)
            consumer_layout = consumer_type.layout.data
            strides = [1]
            for size in reversed(consumer_type.shape.data[1:]):
                strides = [size.data * strides[0]] + strides
            assert isinstance(consumer_type.layout, TiledStridedLayoutAttr)
            tile_bounds = consumer_type.layout.data.tile_bounds()
            producer_layout = TiledStridedLayout.from_strides(strides, tile_bounds)  # pyright: ignore
        else:
            assert isinstance(consumer_type.layout, TiledStridedLayoutAttr)
            consumer_layout = consumer_type.layout.data
            assert isinstance(producer_type.layout, TiledStridedLayoutAttr)
            producer_layout = producer_type.layout.data

        sizes, strides = get_transform(producer_layout, consumer_layout)

        transform_is_null = len(sizes) == 1 and strides == [1]

        # create BDDimlayout
        bd_layout = BDDimLayoutArrayAttr(
            BDDimLayoutArray([BDDimLayout((size, stride)) for size, stride in zip(sizes, strides, strict=True)])
        )

        # take last fifo in the chain (starting form memtile i)
        fifo = SymbolTable.lookup_symbol(device_op, of_name)
        assert isinstance(fifo, ObjectFifoOp)

        # make sure fifo originates from memtile:
        assert isinstance(fifo.producerTile, OpResult) and isinstance(tile_op := fifo.producerTile.op, TileOp)
        if tile_op.row.value.data != 1:
            for link in device_op.walk():
                if isinstance(link, ObjectFifoLinkOp):
                    if of_name in (x.root_reference.data for x in link.fifoIns):
                        fifo = SymbolTable.lookup_symbol(device_op, link.fifoOuts.data[0])
                        assert isinstance(fifo, ObjectFifoOp)

        # fifo.elemType = ObjectFIFO([MemRefType(element_type.element_type, element_type.shape, dest_type.layout)])
        if not transform_is_null:
            fifo.dimensionsToStream = bd_layout

        if consumer_type is not None:
            for consumer in consumers:
                consumer.result.type = ObjectFIFOSubview([consumer_type])
                for use in consumer.result.uses:
                    if isinstance(use.operation, ObjectFIFOSubviewAccessOp):
                        use.operation.output.type = consumer_type

        if producer_type is not None:
            for producer in producers:
                producer.result.type = ObjectFIFOSubview([producer_type])
                for use in producer.result.uses:
                    if isinstance(use.operation, ObjectFIFOSubviewAccessOp):
                        use.operation.output.type = producer_type


@dataclass
class OrderCoreOps(RewritePattern):
    # Complete a bubble-type sorting of core ops for a more deterministic output
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        def get_tile_idx(op: CoreOp) -> tuple[int, int]:
            assert isinstance(op.tile, OpResult)
            assert isinstance(op.tile.op, TileOp)
            return (op.tile.op.col.value.data, op.tile.op.row.value.data)

        if not isinstance(next_op := op.next_op, CoreOp):
            return

        if get_tile_idx(op) > get_tile_idx(next_op):
            next_op.detach()
            rewriter.insert_op(next_op, InsertPoint.before(op))


@dataclass
class InfinteLoopCol(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        aie_end = op.region.block.last_op
        assert isinstance(aie_end, EndOp)
        aie_end.detach()
        start = ConstantOp.from_int_and_width(0, IndexType())
        step = ConstantOp.from_int_and_width(1, IndexType())
        end = ConstantOp.from_int_and_width(0xFFFF_FFFF, IndexType())
        body = rewriter.move_region_contents_to_new_regions(op.region)
        body.block.insert_arg(IndexType(), 0)  # add index argument
        rewriter.insert_op(scf.YieldOp(), InsertPoint.at_end(body.block))
        for_op = ForOp(start, end, step, [], body)
        op.region.add_block(Block([start, step, end, for_op, aie_end]))


@dataclass
class PutEndAtEnd(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, device: DeviceOp, rewriter: PatternRewriter):
        for op in device.walk(reverse=True):
            if isinstance(op, EndOp):
                if op is device.region.block.last_op:
                    # all good
                    return
                else:
                    op.detach()
                    rewriter.insert_op(op, InsertPoint.at_end(device.region.block))
                    return


@dataclass(frozen=True)
class ConvertStreamToAIEPass(ModulePass):
    name = "convert-stream-to-aie"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        assert isinstance(ctx, AIEContext)

        PatternRewriteWalker(OrderCoreOps()).rewrite_module(op)

        PatternRewriteWalker(SetKernelLayouts(ctx.registered_kernels)).rewrite_module(op)
        PatternRewriteWalker(HoistLayoutCasts()).rewrite_module(op)
        PatternRewriteWalker(SquashLayoutCasts()).rewrite_module(op)
        PatternRewriteWalker(ConvertAIEKernels(ctx.registered_kernels)).rewrite_module(op)
        # symbol table stuff messes up my terminator op:
        PatternRewriteWalker(PutEndAtEnd()).rewrite_module(op)

        PatternRewriteWalker(RealizeLayoutCasts()).rewrite_module(op)
        ClearMemorySpace().apply(ctx, op)
        PatternRewriteWalker(InfinteLoopCol(), apply_recursively=False).rewrite_module(op)
