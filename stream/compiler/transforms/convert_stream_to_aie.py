import re
import string
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Self, cast

from snaxc.dialects.snax import LayoutCast
from snaxc.dialects.tsl import TiledStridedLayoutAttr
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.context import MLContext
from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp
from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Attribute, Operation, OpResult, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import (
    AIEDeviceEnum,
    BDDimLayout,
    BDDimLayoutArray,
    BDDimLayoutArrayAttr,
    Block,
    CoreOp,
    DeviceOp,
    EndOp,
    ObjectFIFO,
    ObjectFifoAcquireOp,
    ObjectFifoLinkOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    ObjectFIFOReleaseOp,
    ObjectFIFOSubview,
    ObjectFIFOSubviewAccessOp,
    SymbolTable,
    TileOp,
)
from xdsl_aie.dialects.aiex import (
    DmaMemcpyNdOp,
    DmaWaitOp,
    RuntimeSequenceOp,
)
from zigzag.datatypes import LayerDim

from stream.compiler.dialects.stream import ChannelOp, ComputationNodeOp, EdgeOp, PullOp, PushOp, TransferOp
from stream.compiler.transforms.iteration_space_to_for import iteration_space_to_for
from stream.workload.steady_state.iteration_space import IterationVariableReuse


def get_tile(value: str) -> tuple[int, int]:  # noqa: PLR0911, PLR0912
    if value == "Core(0)":
        return 0, 0
    elif value == "Core(1)":
        return 0, 1
    elif value == "Core(2)":
        return 0, 2
    elif value == "Core(3)":
        return 0, 3
    elif value == "Core(4)":
        return 0, 4
    elif value == "Core(5)":
        return 0, 5
    elif value == "Core(6)":
        return 1, 0
    elif value == "Core(7)":
        return 1, 1
    elif value == "Core(8)":
        return 1, 2
    elif value == "Core(9)":
        return 1, 3
    elif value == "Core(10)":
        return 1, 4
    elif value == "Core(11)":
        return 1, 5
    elif value == "Core(12)":
        return 2, 0
    elif value == "Core(13)":
        return 2, 2
    elif value == "Core(14)":
        return 2, 3
    elif value == "Core(15)":
        return 2, 4
    elif value == "Core(16)":
        return 2, 5
    elif value == "Core(17)":
        return 3, 0
    elif value == "Core(18)":
        return 3, 2
    elif value == "Core(19)":
        return 3, 3
    elif value == "Core(20)":
        return 3, 4
    elif value == "Core(21)":
        return 3, 5

    raise RuntimeError(f"Unknown tile value: {value}")
    match = re.match(r"Core\((\d+)\)", value)
    if match:
        return 0, int(match.group(1))
    else:
        raise ValueError(f"Invalid tile value: {value}")


def get_of_name(source: TileOp, dest: TileOp, operand: str) -> str:
    of_name: str = "of_"
    # compute tile specific objectfifos:
    if source.row.value.data > 1 or dest.row.value.data > 1:
        of_name += f"{source.col.value.data}{source.row.value.data}"
        of_name += f"to{dest.col.value.data}{dest.row.value.data}"
    else:  # shim objectififos
        of_name += f"{source.col.value.data}shim"
    of_name += "_" + operand
    return of_name


@dataclass
class TileOpManager:
    device_op: DeviceOp

    tile_ops: dict[tuple[int, int], TileOp] = field(init=False)

    def __post_init__(self):
        self.tile_ops = {}

        # index existing tile ops
        for op in self.device_op.region.walk():
            if isinstance(op, TileOp):
                self.tile_ops[(op.col.value.data, op.row.value.data)] = op

    def insert_or_update(self, x: int, y: int) -> TileOp:
        # return pre-existing op
        if (x, y) in self.tile_ops:
            return self.tile_ops[(x, y)]

        # create and insert op
        rewriter = Rewriter()
        rewriter.insert_op(tile_op := TileOp(x, y), InsertPoint.at_start(self.device_op.region.block))
        tile_op.result.name_hint = f"tile-{x}-{y}"
        self.tile_ops[(x, y)] = tile_op
        return tile_op

    def get_tile(self, operation: Operation) -> TileOp:
        parent = operation
        while True:
            if isinstance(parent, CoreOp):
                assert isinstance(parent.tile, OpResult) and isinstance(parent.tile.op, TileOp)
                return parent.tile.op
            if isinstance(parent, RuntimeSequenceOp):
                if isinstance(operation, PushOp | PullOp):
                    memtile_idx = get_tile(operation.memtile.data)
                    return self.insert_or_update(memtile_idx[0], 0)
                return self.insert_or_update(0, 0)
            parent = parent.parent_op()
            if parent is None:
                raise RuntimeError()


def is_shim(tile: TileOp) -> bool:
    return tile.row.value.data == 0


@dataclass
class SortPullPushOp:  # noqa: PLW1641 for no hash
    op: PullOp | PushOp

    def __init__(self, op: PullOp | PushOp, tile_op_manager: TileOpManager):
        self.op = op
        self.tile = tile_op_manager.get_tile(op)

    def __lt__(self, other: "SortPullPushOp") -> bool:
        # Compare reversed spatial strides first
        self_strides = list(reversed(self.op.spatial_strides.get_values()))
        other_strides = list(reversed(other.op.spatial_strides.get_values()))

        if self_strides != other_strides:
            return self_strides < other_strides

        # Then compare tile indices: first by row, then by column
        self_col, self_row = self.tile.col.value.data, self.tile.row.value.data
        other_col, other_row = other.tile.col.value.data, other.tile.row.value.data

        return (self_row, self_col) < (other_row, other_col)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SortPullPushOp):
            return False
        return (
            list(reversed(self.op.spatial_strides.get_values()))
            == list(reversed(other.op.spatial_strides.get_values()))
            and self.tile == other.tile
        )


@dataclass
class ObjectFifoHop:
    fifos: list[ObjectFifoOp]

    @property
    def fifo(self) -> ObjectFifoOp:
        assert len(self.fifos) == 1, "More than one fifo in this hop"
        return self.fifos[0]

    @property
    def start(self) -> Iterable[TileOp]:
        for fifo in self.fifos:
            producer = fifo.producerTile
            assert isinstance(producer, OpResult)
            assert isinstance(producer.op, TileOp)
            yield producer.op

    @property
    def end(self) -> Iterable[TileOp]:
        for fifo in self.fifos:
            for consumer in fifo.consumerTiles:
                assert isinstance(consumer, OpResult)
                assert isinstance(consumer.op, TileOp)
                yield consumer.op

    @classmethod
    def to_memtile(
        cls, producers: Sequence[PushOp], memtile: TileOp, tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        # when coming from shim, send to custom handler for memtile reuse
        if is_shim(tile_op_manager.get_tile(producers[0])):
            return cls.shim_to_mem(producers[0], memtile, tile_op_manager, name_base)
        else:
            return cls.compute_to_mem(producers, memtile, tile_op_manager, name_base)

    @classmethod
    def from_memtile(
        cls, consumers: Sequence[PullOp], memtile: TileOp, tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        if is_shim(tile_op_manager.get_tile(consumers[0])):
            return cls.mem_to_shim(consumers[0], memtile, tile_op_manager, name_base)
        else:
            return cls.mem_to_compute(consumers, memtile, tile_op_manager, name_base)

    @classmethod
    def compute_to_mem(
        cls, producers: Sequence[PushOp], memtile: TileOp, tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        assert isinstance(memref_type := producers[0].input.type, MemRefType)
        if len(producers) > 1:
            of_type = "join"
        else:
            of_type = "unicast"
        producers = sorted(producers, key=lambda op: SortPullPushOp(op, tile_op_manager))
        producer_tiles = [tile_op_manager.get_tile(producer) for producer in producers]
        object_fifos: list[ObjectFifoOp] = []
        for i, of_producer in enumerate(producer_tiles):
            if len(producers) > 1:
                of_name = name_base + "_" + of_type + "_" + string.ascii_lowercase[i]
            else:
                of_name = name_base + "_" + of_type
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=IntegerAttr(producers[0].ssis.data.nb_local_tensors_compute(), i32),
                producerTile=of_producer,
                consumerTiles=[memtile],
                referenced_type=memref_type.get_element_type(),
                shape=memref_type.get_shape(),
                name=of_name,
            )
            del object_fifo.properties["repeat_count"]
            object_fifos.append(object_fifo)
        return cls(object_fifos)

    @classmethod
    def mem_to_compute(
        cls, consumers: Sequence[PullOp], memtile: TileOp, tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        assert isinstance(memref_type := consumers[0].output.type, MemRefType)
        distribute = False
        if len(consumers) > 1:
            # determine whether to broadcast / distribute
            distribute = consumers[0].spatial_strides != consumers[1].spatial_strides
            if distribute:
                of_type = "distribute"
            else:
                of_type = "broadcast"
        else:
            of_type = "unicast"
        consumers = sorted(consumers, key=lambda op: SortPullPushOp(op, tile_op_manager))
        consumer_tiles = [tile_op_manager.get_tile(consumer) for consumer in consumers]
        if distribute:
            fifos = [(memtile, [tile]) for tile in consumer_tiles]
        else:
            fifos = [(memtile, consumer_tiles)]
        object_fifos: list[ObjectFifoOp] = []
        for i, (of_producer, of_consumers) in enumerate(fifos):
            if distribute:
                of_name = name_base + "_" + of_type + "_" + string.ascii_lowercase[i]
            else:
                of_name = name_base + "_" + of_type
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=IntegerAttr(consumers[0].ssis.data.nb_local_tensors_compute(), i32),
                producerTile=of_producer,
                consumerTiles=of_consumers,
                referenced_type=memref_type.get_element_type(),
                shape=memref_type.get_shape(),
                name=of_name,
                repeat_count=consumers[0].ssis.data.reuse_factor_mem(),
            )
            assert isinstance(object_fifo.repeat_count, IntegerAttr)
            if object_fifo.repeat_count.value.data == 1:
                del object_fifo.properties["repeat_count"]
            object_fifos.append(object_fifo)
        return cls(object_fifos)

    @classmethod
    def shim_to_mem(cls, producer: PushOp, memtile: TileOp, tile_op_manager: TileOpManager, name_base: str) -> Self:
        assert isinstance(memref_type := producer.input.type, MemRefType)
        spatial_relevant = [
            LayerDim(loop_dim.data)
            for loop_dim, spatial_stride in zip(
                producer.loop_dimensions, producer.spatial_strides.get_values(), strict=True
            )
            if spatial_stride != 0
        ]
        object_fifo = ObjectFifoOp.from_referenced_type(
            elemNumber=IntegerAttr(1, i32),
            producerTile=tile_op_manager.get_tile(producer),
            consumerTiles=[memtile],
            referenced_type=memref_type.get_element_type(),
            shape=producer.ssis.data.shape_mem(spatial_relevant)[::-1]
            + cast(tuple[int, ...], producer.sizes.get_values()),
            name=name_base + "mem",
            repeat_count=1,
        )
        del object_fifo.properties["repeat_count"]

        if object_fifo.repeat_count is not None and object_fifo.repeat_count.value.data == 1:
            del object_fifo.properties["repeat_count"]
        return cls([object_fifo])

    @classmethod
    def mem_to_shim(cls, consumer: PullOp, memtile: TileOp, tile_op_manager: TileOpManager, name_base: str) -> Self:
        assert isinstance(memref_type := consumer.output.type, MemRefType)
        object_fifo = ObjectFifoOp.from_referenced_type(
            elemNumber=IntegerAttr(1, i32),
            producerTile=memtile,
            consumerTiles=[tile_op_manager.get_tile(consumer)],
            referenced_type=memref_type.get_element_type(),
            shape=consumer.ssis.data.shape_mem([])[::-1] + cast(tuple[int, ...], consumer.sizes.get_values()),
            name=name_base + "mem",
        )
        del object_fifo.properties["repeat_count"]
        return cls([object_fifo])


@dataclass
class ObjectFifoChain:
    hops: list[ObjectFifoHop]
    links: list[ObjectFifoLinkOp]

    @property
    def start(self) -> Iterable[TileOp]:
        return self.hops[0].start

    @property
    def end(self) -> Iterable[TileOp]:
        return self.hops[-1].end

    @classmethod
    def from_channel(
        cls, channel: SSAValue, memref_type: MemRefType[Attribute], tile_op_manager: TileOpManager, name_base: str
    ):
        # gather consumers / producers
        producers = list(use.operation for use in channel.uses if isinstance(use.operation, PushOp))
        producer_tiles = [tile_op_manager.get_tile(op) for op in producers]
        consumers = list(use.operation for use in channel.uses if isinstance(use.operation, PullOp))
        consumer_tiles = [tile_op_manager.get_tile(op) for op in consumers]

        # determine hops
        hops: Sequence[ObjectFifoHop]
        if is_shim(consumer_tiles[0]) or is_shim(producer_tiles[0]):
            # pass through the memtile
            memtile = tile_op_manager.insert_or_update(*get_tile(producers[0].memtile.data))
            hops = [
                ObjectFifoHop.to_memtile(producers, memtile, tile_op_manager, name_base),
                ObjectFifoHop.from_memtile(consumers, memtile, tile_op_manager, name_base),
            ]
        else:
            raise NotImplementedError()

        # generate links for every hop
        links: Sequence[ObjectFifoLinkOp] = []
        for i in range(len(hops) - 1):
            link = cls.get_link(hops[i], hops[i + 1])
            links.append(link)

        return cls(hops, links)

    @staticmethod
    def get_link(hop_in: ObjectFifoHop, hop_out: ObjectFifoHop) -> ObjectFifoLinkOp:
        if len(hop_in.fifos) > 1:
            # determine src offsets
            assert isinstance(memref_type := hop_in.fifos[0].elemType.buffer, MemRefType)
            offset = prod(memref_type.get_shape())
            src_offsets = [i * offset for i in range(len(hop_in.fifos))]
        else:
            src_offsets = []
        if len(hop_out.fifos) > 1:
            assert isinstance(memref_type := hop_out.fifos[0].elemType.buffer, MemRefType)
            offset = prod(memref_type.get_shape())
            dst_offsets = [i * offset for i in range(len(hop_out.fifos))]
        else:
            dst_offsets = []
        return ObjectFifoLinkOp(
            [fifo.sym_name.data for fifo in hop_in.fifos],
            [fifo.sym_name.data for fifo in hop_out.fifos],
            src_offsets,
            dst_offsets,
        )

    def get_of(self, op: PullOp | PushOp, tile_op_manager: TileOpManager):
        """
        Get the correct of in this chain for the given operation
        """
        tile = tile_op_manager.get_tile(op)
        for hop in self.hops:
            for fifo in hop.fifos:
                if isinstance(op, PullOp):
                    if tile.result in fifo.consumerTiles:
                        return fifo
                if isinstance(op, PushOp):
                    if tile.result == fifo.producerTile:
                        return fifo
        raise RuntimeError("Fifo not found in chain")

    def __contains__(self, target: ObjectFifoOp) -> bool:
        return any(target in hop.fifos for hop in self.hops)


@dataclass
class ObjectFifoManager:
    tile_op_manager: TileOpManager
    sequence_op: RuntimeSequenceOp
    device_op: DeviceOp

    counter: int = 0
    channel_to_of: dict[SSAValue, ObjectFifoChain] = field(default_factory=dict)

    def insert_or_update(self, channel: SSAValue, memref_type: MemRefType[Attribute]) -> ObjectFifoChain:  # noqa: PLR0915
        # find previous
        if channel in self.channel_to_of:
            return self.channel_to_of[channel]

        of_chain = ObjectFifoChain.from_channel(channel, memref_type, self.tile_op_manager, f"of_{self.counter}")
        self.counter += 1
        self.channel_to_of[channel] = of_chain

        # insert fifo ops
        for hop in of_chain.hops:
            for fifo in hop.fifos:
                SymbolTable.insert_or_update(self.device_op, fifo)

        # insert link ops
        for link in of_chain.links:
            self.device_op.region.block.add_op(link)

        return of_chain

    def get_of_chain(self, of: ObjectFifoOp | str) -> ObjectFifoChain:
        if isinstance(of, str):
            of = self.of_from_name(of)
        for chain in self.channel_to_of.values():
            if of in chain:
                return chain
        raise RuntimeError(f"ObjectFifoOp {of.sym_name.data} not found in channel_to_of mapping")

    def insert_or_update_of(self, object_fifo: ObjectFifoOp) -> ObjectFifoOp:
        SymbolTable.insert_or_update(self.device_op, object_fifo)
        return object_fifo

    def of_from_name(self, name: str) -> ObjectFifoOp:
        result = SymbolTable.lookup_symbol(self.device_op, name)
        assert isinstance(result, ObjectFifoOp)
        return result

    def update_depths(self):
        current_fifo_depth: dict[str, int] = defaultdict(int)

        for op in self.device_op.region.block.walk():
            if isinstance(op, ObjectFifoAcquireOp):
                of_name = op.objFifo_name.root_reference.data

                # update acquire size
                op.size = IntegerAttr.from_int_and_width(current_fifo_depth[of_name] + 1, 32)

                # update access index for all accesses based on this acquire
                for subview_access in [
                    x.operation for x in op.result.uses if isinstance(x.operation, ObjectFIFOSubviewAccessOp)
                ]:
                    subview_access.index = IntegerAttr.from_int_and_width(current_fifo_depth[of_name], 32)

                # increase current_depth
                current_fifo_depth[of_name] += 1

                # increase the depth of objectfifo if it does not suffice
                of = self.of_from_name(of_name)
                if of.elemNumber.value.data < current_fifo_depth[of_name] + 1:
                    of.elemNumber = IntegerAttr.from_int_and_width(current_fifo_depth[of_name] + 1, 32)

            elif isinstance(op, ObjectFIFOReleaseOp):
                of_name = op.objFifo_name.root_reference.data
                current_fifo_depth[of_name] -= 1


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
class PutTransfersBeforeFirstUse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter):
        assert op.parent
        operation_uses = set(x.operation for x in op.results[0].uses)
        try:
            first_use_op: Operation = next(o for o in op.parent.walk() if o in operation_uses)
        except StopIteration:
            # Print descriptive error message with relevant operation uses
            raise RuntimeError(
                f"TransferOp has no uses in the parent region. "
                f"Operation uses: {operation_uses}. "
                f"TransferOp details: {op}."
            ) from None
        while op.parent_op() is not first_use_op.parent_op():
            assert (parent := first_use_op.parent_op()) is not None
            first_use_op = parent

        op.detach()
        rewriter.insert_op(op, InsertPoint.before(first_use_op))


@dataclass
class TransferToRuntimeSequence(RewritePattern):
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        if not isinstance(runtime_sequence := op.parent_op(), RuntimeSequenceOp):
            return

        if isinstance(op, PushOp):
            memref_type = op.input.type
        else:
            memref_type = op.output.type
        assert isinstance(memref_type, MemRefType)
        of_chain = self.object_fifo_manager.insert_or_update(op.channel, memref_type)

        # get edge
        if isinstance(op, PushOp):
            assert isinstance(op.input, OpResult)
            edge = op.input.op
            of = of_chain.get_of(op, self.object_fifo_manager.tile_op_manager)
        else:
            edge = next((use.operation for use in op.output.uses), None)
            if edge is None:
                # TODO: this transfer should not be present anymore
                rewriter.erase_matched_op()
                return
            of = of_chain.get_of(op, self.object_fifo_manager.tile_op_manager)
        assert isinstance(edge, EdgeOp)

        of_name = of.sym_name.data

        arg_order = ["Op0.I_in", "Op0.W_in", "Op0.O_out"]

        arg_index = arg_order.index(edge.tensor.data)
        arg = runtime_sequence.body.block.args[arg_index]

        # offsets = cast(tuple[int, ...], op.offsets.get_values()[-4:])
        sizes = cast(tuple[int, ...], op.sizes.get_values()[-4:])
        strides = cast(tuple[int, ...], op.strides.get_values()[-4:])
        offsets = cast(tuple[int, ...], op.offsets.get_values()[-4:])
        assert isinstance(arg.type, MemRefType)
        shapes = tuple(x.data for x in arg.type.shape)[-4:]

        # assume default layout here:
        static_strides = []
        current_stride = 1
        total_offset = 0
        for shape, stride, offset in zip(reversed(shapes), reversed(strides), reversed(offsets), strict=False):
            static_strides.insert(0, current_stride)
            total_offset += current_stride * offset
            current_stride *= shape * stride

        static_sizes = list(sizes)

        static_sizes, static_strides = canonicalize_transformation(static_sizes, static_strides)

        # add the repeating pattern
        # offset is definitely zero for now
        for iter_var in op.ssis.data.variables:
            loop_dimensions = [x.data for x in op.loop_dimensions]
            if (iter_var.relevant or IterationVariableReuse.MEM_TILE_NO_REUSE in iter_var.reuse) and iter_var.size > 1:
                # sptial dims must only emit extra sizes and strides if they are used as spatial strides
                if iter_var.spatial:
                    index = loop_dimensions.index(str(iter_var.dimension))
                    if op.spatial_strides.get_values()[index] == 0:
                        continue
                if str(iter_var.dimension) in loop_dimensions:
                    index = loop_dimensions.index(str(iter_var.dimension))
                    stride = prod(memref_type.get_shape()[index + 1 :]) * op.sizes.get_values()[index]
                    # stride = prod(op.sizes.get_values()[index:])
                    assert isinstance(stride, int)
                else:
                    # repeat
                    stride = 0
                static_sizes.insert(0, iter_var.size)
                static_strides.insert(0, stride)

        # canonicalize transformation
        # static_sizes, static_strides = canonicalize_transformation(static_sizes, static_strides)
        MAX_STATIC_SIZE_LEN = 5
        if len(static_sizes) > MAX_STATIC_SIZE_LEN:
            raise RuntimeError()
        if len(static_sizes) == MAX_STATIC_SIZE_LEN:
            software_size = static_sizes.pop(0)
            software_stride = static_strides.pop(0)
        else:
            software_size = 1
            software_stride = 0

        static_sizes = (1,) * (4 - len(static_sizes)) + tuple(static_sizes)
        static_strides = (0,) * (4 - len(static_strides)) + tuple(static_strides)

        id = int(of_name[3])
        for i in range(software_size):
            software_offset = i * software_stride
            total_offset += software_offset
            static_offsets = (0, 0, 0, total_offset)
            # Insert DMA
            memcpy = DmaMemcpyNdOp(
                arg,
                static_offsets=static_offsets,
                static_sizes=static_sizes,
                static_strides=static_strides,
                metadata=of_name,
                id=id,
                issue_token=True,
            )
            rewriter.insert_op(memcpy, InsertPoint.before(op))

        # remove output from edge op operands
        rewriter.erase_matched_op(safe_erase=False)


@dataclass
class TransferToObjectFIFOPattern(RewritePattern):
    object_fifo_manager: ObjectFifoManager

    release_op: dict[str, Operation | None] = field(default_factory=dict)  # pyright: ignore

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        # do the runtime sequence thing elsewhere, must have a core_op parent
        parent = op
        while True:
            if isinstance(parent, CoreOp):
                break
            parent = parent.parent_op()
            if parent is None:
                return

        if isinstance(op, PushOp):
            memref_type = op.input.type
        else:
            memref_type = op.output.type
        assert isinstance(memref_type, MemRefType)
        of_chain = self.object_fifo_manager.insert_or_update(op.channel, memref_type)

        # get spatial offset to fetch from correct object fifo
        offset = 0
        for size, spatial_stride in zip(op.sizes.get_values(), op.spatial_strides.get_values(), strict=True):
            if spatial_stride != 0:
                offset += int((spatial_stride // size) * prod(op.sizes.get_values()))

        # decide whether to consume or produce
        if isinstance(op, PullOp):
            port = ObjectFifoPortEnum.Consume
            of = of_chain.get_of(op, self.object_fifo_manager.tile_op_manager)

            operand = op.output
        else:
            port = ObjectFifoPortEnum.Produce
            of = of_chain.get_of(op, self.object_fifo_manager.tile_op_manager)
            operand = op.input

        of_name = of.sym_name.data

        assert isinstance(memref_type := operand.type, MemRefType)

        first_relevant_iter = next(iv for iv in op.ssis.data.get_temporal_variables() if iv.relevant)
        first_relevant_index = op.ssis.data.get_temporal_variables().index(first_relevant_iter)

        last_reuse = next(
            (
                iv
                for iv in op.ssis.data.get_temporal_variables()[::-1]
                if IterationVariableReuse.COMPUTE_TILE_REUSE in iv.reuse
            ),
            None,
        )
        if last_reuse:
            last_reuse_index = op.ssis.data.get_temporal_variables().index(last_reuse)
            reuse_iters = op.ssis.data.get_temporal_variables()[first_relevant_index : last_reuse_index + 1]
        else:
            reuse_iters = []

        relevant_reuse_iters = [iv for iv in reuse_iters if iv.relevant]

        reuse_factor = prod(iv.size for iv in reuse_iters if iv.relevant)

        # update object fifo depth
        # of.elemNumber = IntegerAttr.from_int_and_width(reuse_factor, 32)

        # acquire:
        acquire_op = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(reuse_factor, 32),
            object_fifo=of_name,
            shape=memref_type.get_shape(),
            element_type=memref_type.get_element_type(),
        )

        # accesses:
        access_ops = [ObjectFIFOSubviewAccessOp(IntegerAttr(i, i32), acquire_op) for i in range(reuse_factor)]

        # index op to select correct access:
        index_ops: list[Operation] = [
            mult_val := ConstantOp.from_int_and_width(1, IndexType()),
            add_val := ConstantOp.from_int_and_width(0, IndexType()),
        ]
        for_op = op.parent_op()
        assert isinstance(for_op, ForOp)
        for iter_var in relevant_reuse_iters:
            assert "layer_dim" in for_op.attributes
            while for_op.attributes["layer_dim"] != StringAttr(iter_var.dimension.name):
                for_op = for_op.parent_op()
                assert isinstance(for_op, ForOp)
            i_arg = MuliOp(mult_val, for_op.body.block.args[0])
            add_val = AddiOp(add_val, i_arg)
            mult_val = MuliOp(mult_val, for_op.ub)
            index_ops.extend([i_arg, add_val, mult_val])

        index_switch = IndexSwitchOp(
            arg=add_val,
            cases=DenseArrayBase.from_list(IntegerType(64), list(range(reuse_factor))),
            default_region=Region(Block([YieldOp(access_ops[0])])),
            case_regions=[Region(Block([YieldOp(access_ops[i])])) for i in range(reuse_factor)],
            result_types=access_ops[0].result_types,
        )
        index_ops.append(index_switch)

        release_op = ObjectFIFOReleaseOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(reuse_factor, 32),
            object_fifo=of_name,
        )

        # there should only be one use now
        if isinstance(op, PullOp):
            use_op = next(use.operation for use in op.output.uses)
        else:
            assert isinstance(op.input, OpResult)
            use_op = op.input.op

        # get to same level in the iteration tree:
        while op.parent is not use_op.parent:
            use_op = use_op.parent_op()
            assert use_op is not None

        # put acquire and accesses at last level of reuse:
        use_op_reuse = use_op
        for _ in range(len(reuse_iters)):
            use_op_reuse = use_op_reuse.parent_op()
            assert use_op_reuse is not None

        rewriter.insert_op(release_op, InsertPoint.after(use_op_reuse))
        rewriter.insert_op([acquire_op, *access_ops], InsertPoint.before(use_op_reuse))
        rewriter.insert_op(index_ops, InsertPoint.before(use_op))

        # set output of computation node op if this was a push op
        if isinstance(op, PushOp):
            assert isinstance(op.input, OpResult)
            assert isinstance(compute := op.input.op, ComputationNodeOp)
            new_compute = ComputationNodeOp(
                compute.inputs,
                index_switch.results[0],
                compute.kernel.data,
                compute.core_allocation.data,
                compute.ssis.data,
                compute.result_types,
            )
            rewriter.replace_op(compute, new_compute)

        operand.replace_by(index_switch.results[0])
        rewriter.erase_matched_op()

        return


@dataclass
class MMPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "mm_32x32x32":
            return

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        function_name = "matmul_i16_i32"

        func_op = FuncOp(function_name, (input_types, []), Region(), "private")
        zero_func_op = FuncOp("zero_i32", (input_types[-1:], []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)
        SymbolTable.insert_or_update(device_op, zero_func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr(op.kernel.data + ".o")

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)

        # insert zero func call for first use
        output = SSAValue.get(inputs[-1])
        assert isinstance(output, OpResult)
        zero_call = CallOp("zero_i32", inputs[-1:], [])
        rewriter.insert_op(zero_call, InsertPoint.after(output.op))

        func_call = CallOp(function_name, inputs, [])
        rewriter.insert_op(func_call, InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class ConvPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "conv2dk1_i8":
            return

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        # four i32's?
        input_types.extend([i32] * 4)

        func_op = FuncOp(op.kernel.data, (input_types, []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr(op.kernel.data + ".o")

        c32 = ConstantOp.from_int_and_width(32, i32)
        c64 = ConstantOp.from_int_and_width(64, i32)
        c10 = ConstantOp.from_int_and_width(10, i32)

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)
        inputs.extend([c32, c64, c64, c10])

        func_call = CallOp(op.kernel.data, inputs, [])

        rewriter.replace_matched_op((c32, c64, c10, func_call))


@dataclass
class PassThroughMemTile(RewritePattern):
    changes: dict[str, str]
    tile_op_manager: TileOpManager
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ObjectFifoOp, rewriter: PatternRewriter):
        # not supporting any broadcast yet
        if len(op.consumerTiles) != 1:
            return

        # if connects to shim:
        assert isinstance(producerTile := op.producerTile, OpResult)
        assert isinstance(producerTile.op, TileOp)
        assert isinstance(consumerTile := op.consumerTiles[0], OpResult)
        assert isinstance(consumerTile.op, TileOp)

        # source/dest must be shim
        if producerTile.op.row.value.data == 0:
            # shim = producerTile
            compute = consumerTile
            shim_is_producer = True
        elif consumerTile.op.row.value.data == 0:
            # shim = consumerTile
            compute = producerTile
            shim_is_producer = False
        else:
            return

        # other one must be compute tile
        assert isinstance(compute.op, TileOp)
        if compute.op.row.value.data < 2:  # noqa: PLR2004
            return

        memtile = self.tile_op_manager.insert_or_update(0, 1)

        objectfifo_compute = ObjectFifoOp(
            memtile if shim_is_producer else op.producerTile,
            list(op.consumerTiles) if shim_is_producer else [memtile],
            op.elemNumber,
            op.elemType,
            op.sym_name,
            op.dimensionsToStream,
            op.dimensionsFromStreamPerConsumer,
            op.disable_synchronization,
            op.plio,
            op.via_DMA,
        )

        operand = op.sym_name.data.split("_")[-1]
        if shim_is_producer:
            shim_name = get_of_name(producerTile.op, memtile, operand)
        else:
            shim_name = get_of_name(memtile, consumerTile.op, operand)

        objectfifo_shim = ObjectFifoOp(
            op.producerTile if shim_is_producer else memtile,
            [memtile] if shim_is_producer else list(op.consumerTiles),
            op.elemNumber,
            op.elemType,
            shim_name,
            op.dimensionsToStream,
            op.dimensionsFromStreamPerConsumer,
            op.disable_synchronization,
            op.plio,
            op.via_DMA,
        )

        self.changes[op.sym_name.data] = shim_name

        self.object_fifo_manager.insert_or_update_of(objectfifo_compute)
        self.object_fifo_manager.insert_or_update_of(objectfifo_shim)

        if shim_is_producer:
            link = ObjectFifoLinkOp([shim_name], [op.sym_name.data], [], [])
        else:
            link = ObjectFifoLinkOp([op.sym_name.data], [shim_name], [], [])

        rewriter.insert_op(link, InsertPoint.after(objectfifo_shim))


@dataclass
class SetDistribution(RewritePattern):
    runtime_sequence: RuntimeSequenceOp
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, device_op: DeviceOp, rewriter: PatternRewriter):
        of_links: dict[SymbolRefAttr, list[SymbolRefAttr]] = defaultdict(list)
        of_link_ops: dict[tuple[SymbolRefAttr, SymbolRefAttr], ObjectFifoLinkOp] = {}

        for op in device_op.region.block.ops:
            if isinstance(op, ObjectFifoLinkOp):
                if len(op.fifoIns) != 1 or len(op.fifoOuts) != 1:
                    continue
                of_links[op.fifoIns.data[0]].append(op.fifoOuts.data[0])
                of_link_ops[(op.fifoIns.data[0], op.fifoOuts.data[0])] = op

        # filter out sets with only one link, no distribute needed
        of_links = {
            source: dests
            for source, dests in of_links.items()
            if len(dests) > 1 and "shim" in source.root_reference.data
        }
        # otherwise, sort values
        of_links = {
            source: sorted(dests, key=lambda dest: dest.root_reference.data) for source, dests in of_links.items()
        }

        # order the copies
        for source, dests in of_links.items():
            # list of copies mapping destination to list of copies
            copies: dict[SymbolRefAttr, list[DmaMemcpyNdOp]] = {}
            for dest in dests:
                copies[dest] = []

            for op in self.runtime_sequence.walk():
                if not isinstance(op, DmaMemcpyNdOp):
                    continue
                if op.metadata in copies:
                    copies[op.metadata].append(op)

            # for a correct distribute pattern, all elements should copy the same number of elements
            lengths = [len(v) for v in copies.values()]
            if len(set(lengths)) != 1:
                raise RuntimeError("distribute pattern detected with differing number of dma copies")

            # reorder memcpys based on the first root reference
            for i in range(lengths[0]):
                op = copies[dests[0]][i]
                for j in range(1, len(dests)):
                    new_op = copies[dests[j]][i]
                    new_op.detach()
                    rewriter.insert_op(new_op, InsertPoint.after(op))
                    op = new_op

            # create link op
            # calculate destination offset
            of_source = self.object_fifo_manager.of_from_name(source.root_reference.data)
            assert isinstance(memref_type := of_source.elemType.buffer, MemRefType)
            nb_elements = prod(memref_type.get_shape())
            dst_offsets = list(range(0, nb_elements * len(dests), nb_elements))

            # update source object fifo shape
            of_source.elemType = ObjectFIFO.from_element_type_and_shape(
                memref_type.get_element_type(), (len(dests),) + memref_type.get_shape()
            )

            # create new link op
            new_link_op = ObjectFifoLinkOp([source], dests, [], dst_offsets)

            # insert after last link
            rewriter.insert_op(new_link_op, InsertPoint.after(of_link_ops[(source, dests[-1])]))

            # erase all the rest:
            for i in range(len(dests)):
                rewriter.erase_op(of_link_ops[(source, dests[i])])


@dataclass
class SetJoin(RewritePattern):
    runtime_sequence: RuntimeSequenceOp
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, device_op: DeviceOp, rewriter: PatternRewriter):
        of_links: dict[SymbolRefAttr, list[SymbolRefAttr]] = defaultdict(list)
        of_link_ops: dict[tuple[SymbolRefAttr, SymbolRefAttr], ObjectFifoLinkOp] = {}

        for op in device_op.region.block.ops:
            if isinstance(op, ObjectFifoLinkOp):
                if len(op.fifoIns) != 1 or len(op.fifoOuts) != 1:
                    continue
                of_links[op.fifoOuts.data[0]].append(op.fifoIns.data[0])
                of_link_ops[(op.fifoOuts.data[0], op.fifoIns.data[0])] = op

        # filter out sets with only one link, no distribute needed
        of_links = {
            dest: sources
            for dest, sources in of_links.items()
            if len(sources) > 1 and "shim" in dest.root_reference.data
        }
        # otherwise, sort values
        of_links = {
            dest: sorted(sources, key=lambda source: source.root_reference.data) for dest, sources in of_links.items()
        }

        # order the copies
        for dest, sources in of_links.items():
            # list of copies mapping destination to list of copies
            copies: dict[SymbolRefAttr, list[DmaMemcpyNdOp]] = {}
            for source in sources:
                copies[source] = []

            for op in self.runtime_sequence.walk():
                if not isinstance(op, DmaMemcpyNdOp):
                    continue
                if op.metadata in copies:
                    copies[op.metadata].append(op)

            # for a correct distribute pattern, all elements should copy the same number of elements
            lengths = [len(v) for v in copies.values()]
            if len(set(lengths)) != 1:
                raise RuntimeError("join pattern detected with differing number of dma copies")

            # reorder memcpys based on the first root reference
            for i in range(lengths[0]):
                op = copies[sources[0]][i]
                for j in range(1, len(sources)):
                    new_op = copies[sources[j]][i]
                    new_op.detach()
                    rewriter.insert_op(new_op, InsertPoint.after(op))
                    op = new_op

            # create link op
            # calculate destination offset
            of_dest = self.object_fifo_manager.of_from_name(dest.root_reference.data)
            assert isinstance(memref_type := of_dest.elemType.buffer, MemRefType)
            nb_elements = prod(memref_type.get_shape())
            src_offsets = list(range(0, nb_elements * len(sources), nb_elements))

            # update dest object fifo shape
            of_dest.elemType = ObjectFIFO.from_element_type_and_shape(
                memref_type.get_element_type(), (len(sources),) + memref_type.get_shape()
            )

            # create new link op
            new_link_op = ObjectFifoLinkOp(sources, [dest], src_offsets, [])

            # insert after last link
            rewriter.insert_op(new_link_op, InsertPoint.after(of_link_ops[(dest, sources[-1])]))

            # erase all the rest:
            for i in range(len(sources)):
                rewriter.erase_op(of_link_ops[(dest, sources[i])])


def simplify_strides(
    strides: tuple[int, ...], sizes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """
    Simplify strides. If possible, collapse two dimensions with the same stride into one.
    If not possible, return None.
    """

    if strides[0] == 0 and sizes[0] != 1:
        # special handling for repeat dma copies
        if strides[1] == 0 and sizes[1] != 1:
            strides = (0,) + strides[2:]
            sizes = (sizes[0] * sizes[1],) + sizes[2:]
            return sizes, strides
        else:
            return None
    same_strides = [strides[i] == strides[i + 1] * sizes[i + 1] for i in range(len(strides) - 1)]
    if True in same_strides:
        collapse_idx = same_strides.index(True)
        new_size = sizes[collapse_idx] * sizes[collapse_idx + 1]
        new_stride = strides[collapse_idx + 1]
        sizes = sizes[:collapse_idx] + (new_size,) + sizes[collapse_idx + 2 :]
        strides = strides[:collapse_idx] + (new_stride,) + strides[collapse_idx + 2 :]
        return sizes, strides


@dataclass
class CollapseMemcpys(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DmaMemcpyNdOp, rewriter: PatternRewriter):
        # gather offsets, sizes and strides
        offset_1 = cast(tuple[int, ...], op.static_offsets.get_values())[-1]
        sizes_1 = cast(tuple[int, ...], op.static_sizes.get_values())
        strides_1 = cast(tuple[int, ...], op.static_strides.get_values())

        first_non_1 = next((i for i, x in enumerate(sizes_1) if x != 1), len(sizes_1))
        sizes_1 = sizes_1[first_non_1:]
        strides_1 = strides_1[first_non_1:]

        # check if we can simplify
        simplified = simplify_strides(strides_1, sizes_1)
        if simplified is not None:
            sizes_1 = (1,) * (4 - len(simplified[0])) + simplified[0]
            strides_1 = (0,) * (4 - len(simplified[1])) + simplified[1]

            new_op = DmaMemcpyNdOp(
                op.memref,
                op.static_offsets,
                sizes_1,
                strides_1,
                op.metadata,
                op.id,
                op.issue_token,
                op.offsets,
                op.strides,
            )
            rewriter.replace_matched_op(new_op)
            return

        # find next memcpy with the same metadata
        next_op = op
        while True:
            next_op = next_op.next_op
            if next_op is None:
                return
            if isinstance(next_op, DmaMemcpyNdOp) and next_op.metadata == op.metadata:
                break

        # strides should fully overlap
        offset_2 = cast(tuple[int, ...], next_op.static_offsets.get_values())[-1]
        sizes_2 = cast(tuple[int, ...], next_op.static_sizes.get_values())
        strides_2 = cast(tuple[int, ...], next_op.static_strides.get_values())

        first_non_1 = next((i for i, x in enumerate(sizes_2) if x != 1), len(sizes_2))
        sizes_2 = sizes_2[first_non_1:]
        strides_2 = strides_2[first_non_1:]

        # full overlap:
        if sizes_1 == sizes_2 and strides_1 == strides_2:
            if (offset_2 - offset_1) == 0:
                # special case as only 4th can be zero
                sizes_1 = (1,) * (3 - len(sizes_1)) + sizes_1
                strides_1 = (0,) * (3 - len(strides_1)) + strides_1
            sizes_1 = (2,) + sizes_1
            strides_1 = (offset_2 - offset_1,) + strides_1

            simplified = simplify_strides(strides_1, sizes_1)
            if simplified is not None:
                sizes_1 = simplified[0]
                strides_1 = simplified[1]

            if len(sizes_1) > 4:  # noqa: PLR2004
                return
            sizes_1 = (1,) * (4 - len(sizes_1)) + sizes_1
            strides_1 = (0,) * (4 - len(strides_1)) + strides_1
            # remove next op
            new_op = DmaMemcpyNdOp(
                op.memref,
                op.static_offsets,
                sizes_1,
                strides_1,
                op.metadata,
                op.id,
                op.issue_token,
                op.offsets,
                op.strides,
            )
            rewriter.replace_matched_op(new_op)
            rewriter.erase_op(next_op)


@dataclass
class OfNameRewriter(RewritePattern):
    changes: dict[str, str]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DmaMemcpyNdOp, rewriter: PatternRewriter):
        if op.metadata.root_reference.data in self.changes:
            op.metadata = SymbolRefAttr(self.changes[op.metadata.root_reference.data])


@dataclass
class InsertRuntimeDMAs(RewritePattern):
    sequence_op: RuntimeSequenceOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ObjectFifoOp, rewriter: PatternRewriter):
        # Add Block Argument to SequenceOp
        memref_type = op.elemType.buffer

        assert isinstance(memref_type, MemRefType)

        shape = list(memref_type.get_shape())
        memref_type = MemRefType(memref_type.get_element_type(), shape)

        sequence_block = self.sequence_op.body.block

        sequence_block.insert_arg(memref_type, 0)

        # Insert DMA
        memcpy = DmaMemcpyNdOp(
            sequence_block.args[0],
            static_offsets=[0, 0, 0, 0],
            static_sizes=[1, 1, 1, memref_type.get_shape()[0]],
            static_strides=[0, 0, 0, 1],
            metadata=op.sym_name,
            id=0,
            issue_token=True,
        )

        rewriter.insert_op(memcpy, InsertPoint.at_start(sequence_block))

        # wait for it ...

        wait = DmaWaitOp(op.sym_name)

        rewriter.insert_op(wait, InsertPoint.at_end(sequence_block))


@dataclass
class EraseEdges(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EdgeOp | ChannelOp, rewriter: PatternRewriter) -> None:
        rewriter.erase_matched_op()


@dataclass
class ManageSyncs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RuntimeSequenceOp, rewriter: PatternRewriter):
        active_ids: set[str] = set()

        for memcpy in op.walk():
            if not isinstance(memcpy, DmaMemcpyNdOp):
                continue

            symbol = memcpy.metadata.string_value()

            if symbol in active_ids:
                rewriter.insert_op(DmaWaitOp(symbol), InsertPoint.before(memcpy))

            active_ids.add(symbol)

        for symbol in active_ids:
            rewriter.insert_op(DmaWaitOp(symbol), InsertPoint.at_end(op.body.block))


@dataclass
class OptimizeWaits(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, runtime: RuntimeSequenceOp, rewriter: PatternRewriter):
        last_wait = None
        for op in runtime.walk():
            if isinstance(op, DmaWaitOp):
                last_wait = op
            if last_wait is not None and isinstance(op, DmaMemcpyNdOp) and op.metadata != last_wait.symbol:
                op.detach()
                rewriter.insert_op(op, InsertPoint.before(last_wait))
                return


@dataclass
class SetKernelLayouts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        # handle the conv case
        if op.callee.root_reference.data == "conv2dk1_i8":
            input = op.arguments[0]
            output = op.arguments[2]
            input_type = cast(MemRefType[Attribute], op.arguments[0].type)

            if isinstance(input_type.layout, TiledStridedLayoutAttr):
                return

            input_layout = TiledStridedLayout(
                [
                    TiledStride([Stride(32 * 64, 1)]),  # N
                    TiledStride([Stride(32 * 64, 1)]),  # G
                    TiledStride([Stride(32 * 64, 1)]),  # H
                    TiledStride([Stride(8, 32)]),  # W
                    TiledStride([Stride(8 * 32, 8), Stride(1, 8)]),  # C
                ]
            )

            input_type = MemRefType(
                input_type.element_type, input_type.shape, TiledStridedLayoutAttr(input_layout), input_type.memory_space
            )

            new_input = LayoutCast(input, input_type)
            new_output = LayoutCast(output, input_type)

            rewriter.insert_op([new_input, new_output], InsertPoint.before(op))

            op.operands[0] = new_input.results[0]
            op.operands[2] = new_output.results[0]

        if op.callee.root_reference.data == "matmul_i16_i32":
            A_operand = op.operands[0]
            A_type = cast(MemRefType[Attribute], op.arguments[0].type)
            if isinstance(A_type.layout, TiledStridedLayoutAttr):
                return
            layout_A = TiledStridedLayout(
                [
                    TiledStride([Stride(16 * 32 // 4, 32 // 4), Stride(4, 4)]),
                    TiledStride([Stride(16, 32 // 4), Stride(1, 4)]),
                ]
            )
            A_type_new = MemRefType(
                A_type.element_type, A_type.shape, TiledStridedLayoutAttr(layout_A), A_type.memory_space
            )
            A_new = LayoutCast(A_operand, A_type_new)

            B_operand = op.operands[1]
            B_type = cast(MemRefType[Attribute], op.arguments[1].type)
            if isinstance(B_type.layout, TiledStridedLayoutAttr):
                return
            layout_B = TiledStridedLayout(
                [
                    TiledStride([Stride(16 * 32 // 4, 32 // 4), Stride(4, 4)]),
                    TiledStride([Stride(16, 32 // 4), Stride(1, 4)]),
                ]
            )
            B_type_new = MemRefType(
                B_type.element_type, B_type.shape, TiledStridedLayoutAttr(layout_B), B_type.memory_space
            )
            B_new = LayoutCast(B_operand, B_type_new)

            D_operand = op.operands[2]
            D_type = cast(MemRefType[Attribute], op.arguments[2].type)
            if isinstance(D_type.layout, TiledStridedLayoutAttr):
                return
            layout_D = TiledStridedLayout(
                [
                    TiledStride([Stride(16 * 32 // 4, 32 // 4), Stride(4, 4)]),
                    TiledStride([Stride(16, 32 // 4), Stride(1, 4)]),
                ]
            )
            D_type_new = MemRefType(
                D_type.element_type, D_type.shape, TiledStridedLayoutAttr(layout_D), D_type.memory_space
            )
            D_new = LayoutCast(D_operand, D_type_new)

            rewriter.insert_op((A_new, B_new, D_new), InsertPoint.before(op))

            op.operands[0] = A_new.results[0]
            op.operands[1] = B_new.results[0]
            op.operands[2] = D_new.results[0]


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
    sizes_dest, strides_dest = zip(*[(x["stride_dest"].bound, x["stride_dest"].step) for x in strides], strict=False)

    # canonicalize
    sizes_src, strides_src = canonicalize_transformation(sizes_src, strides_src)
    sizes_dest, strides_dest = canonicalize_transformation(sizes_dest, strides_dest)

    # we only consider transformations at the source for now, so no transform should be happening at dest
    if len(sizes_dest) != 1:
        raise RuntimeError("did not expect dest transformation")

    return (sizes_src, strides_src)


@dataclass
class RealizeLayoutCats(RewritePattern):
    of_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter):
        # gather some variables
        assert isinstance(op.source, OpResult)
        assert isinstance(index_op := op.source.op, IndexSwitchOp)
        assert isinstance(
            subview_access := index_op.default_region.block.last_op.operands[0].op, ObjectFIFOSubviewAccessOp
        )
        # assert isinstance(subview_access := op.source.op, ObjectFIFOSubviewAccessOp)
        assert isinstance(subview_access.subview, OpResult)
        assert isinstance(of_acquire := subview_access.subview.op, ObjectFifoAcquireOp)

        dest_type = cast(MemRefType[Attribute], op.dest.type)

        # get the objectfifo
        # check if producer or consumer
        port = ObjectFifoPortEnum.from_int(of_acquire.port.value.data)

        # get the chain:
        chain = self.of_manager.get_of_chain(of_acquire.objFifo_name.root_reference.data)

        # get the hop starting from a memtile:
        hop = chain.hops[1]

        # get the element_type
        element_type = cast(MemRefType[Attribute], hop.fifos[0].elemType.buffer)

        tsl_dest = cast(TiledStridedLayoutAttr, dest_type.layout).data

        # create default tsl layout for source:
        strides = [1]
        for size in reversed(dest_type.shape.data[1:]):
            strides = [size.data * strides[0]] + strides
        tile_bounds = tsl_dest.tile_bounds()

        tsl_in = TiledStridedLayout.from_strides(strides, tile_bounds)  # pyright: ignore
        tsl_out = cast(TiledStridedLayoutAttr, dest_type.layout).data

        # calculate transform

        # check if producer on consumer
        if port == ObjectFifoPortEnum.Consume:
            sizes, strides = get_transform(tsl_in, tsl_out)
        else:  # Produce
            sizes, strides = get_transform(tsl_out, tsl_in)

        # create BDDimlayout
        bd_layout = BDDimLayoutArrayAttr(
            BDDimLayoutArray([BDDimLayout((size, stride)) for size, stride in zip(sizes, strides, strict=False)])
        )

        for fifo in hop.fifos:
            fifo.elemType = ObjectFIFO([MemRefType(element_type.element_type, element_type.shape, dest_type.layout)])
            fifo.dimensionsToStream = bd_layout

        element_type = cast(MemRefType[Attribute], hop.fifos[0].elemType.buffer)

        of_layout = element_type.layout

        assert of_layout == dest_type.layout
        # transform has already been applied to ObjectFIFO
        of_acquire.results[0].type = ObjectFIFOSubview([dest_type])
        subview_access.results[0].type = dest_type
        index_op.results[0].type = dest_type
        assert op.source.type == op.dest.type
        op.dest.replace_by(op.source)
        rewriter.erase_matched_op()


@dataclass
class WrapInCoreOps(RewritePattern):
    tile_op_manager: TileOpManager
    of_manager: ObjectFifoManager
    core_ops: dict[tuple[int, int], CoreOp | RuntimeSequenceOp]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):  # noqa: PLR0912
        shim = (0, 0)
        if isinstance(op, ComputationNodeOp):
            core = get_tile(op.core_allocation.data)
        elif isinstance(op, EdgeOp):
            core = shim
        elif isinstance(op, PushOp):
            # what am i pushing?
            assert isinstance(op.input, OpResult)
            if isinstance(op.input.op, EdgeOp):
                core = shim
            elif isinstance(op.input.op, ComputationNodeOp):
                core = get_tile(op.input.op.core_allocation.data)
            else:
                raise NotImplementedError()
        elif isinstance(op, PullOp):
            # where am i pulling to?
            assert len(op.output.uses) == 1
            use = next(iter(op.output.uses))
            if isinstance(use.operation, EdgeOp):
                core = shim
            elif isinstance(use.operation, ComputationNodeOp):
                core = get_tile(use.operation.core_allocation.data)
            else:
                raise NotImplementedError()
        else:
            return

        # create core op if it doesn't exist yet
        if core not in self.core_ops:
            core_op = CoreOp(None, self.tile_op_manager.insert_or_update(*core), Region(Block([EndOp()])))
            rewriter.insert_op(core_op, InsertPoint.at_end(self.tile_op_manager.device_op.region.block))
            self.core_ops[core] = core_op
        else:
            core_op = self.core_ops[core]

        op.detach()
        if isinstance(core_op, CoreOp):
            assert core_op.region.block.last_op
            insert_point = InsertPoint.before(core_op.region.block.last_op)
        else:
            insert_point = InsertPoint.at_end(core_op.body.block)
        rewriter.insert_op(op, insert_point)


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
        for_op = ForOp(start, end, step, [], body)
        op.region.add_block(Block([start, step, end, for_op, aie_end]))


class ConvertStreamToAIEPass(ModulePass):
    name = "convert-stream-to-aie"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # wrap everything in a device op
        #

        rewriter = Rewriter()
        device_op = DeviceOp(
            IntegerAttr.from_int_and_width(AIEDeviceEnum.npu1.get_int(), 32),
            rewriter.move_region_contents_to_new_regions(op.body),
        )
        op.body.add_block(Block([device_op]))

        # add a runtime sequence operation
        # find all edges
        edges: list[EdgeOp] = [edge for edge in op.walk() if isinstance(edge, EdgeOp)]
        order = ["Op0.I_in", "Op0.W_in", "Op0.O_out"]

        runtime_arg_types = []
        for operand_name in order:
            edge = next(edge for edge in edges if edge.tensor.data == operand_name)
            operand = edge.inputs[0] if len(edge.inputs) else edge.output
            assert operand is not None
            runtime_arg_types.append(operand.type)

        runtime_sequence = RuntimeSequenceOp(Region(Block(arg_types=runtime_arg_types)))
        rewriter.insert_op(runtime_sequence, InsertPoint.at_end(device_op.region.block))

        tile_op_manager = TileOpManager(device_op)
        object_fifo_manager = ObjectFifoManager(tile_op_manager, runtime_sequence, device_op)

        # Order all transfers based on first use
        # PatternRewriteWalker(PutTransfersBeforeFirstUse(), apply_recursively=False).rewrite_module(op)
        #

        PatternRewriteWalker(
            WrapInCoreOps(
                tile_op_manager,
                object_fifo_manager,
                {
                    (0, 0): runtime_sequence,
                },
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        for core_op in device_op.region.block.ops:
            if isinstance(core_op, CoreOp):
                # insert runtime sequence op
                iteration_space_to_for(core_op.region.block, rewriter)

        # Convert transfers to object fifo patterns
        PatternRewriteWalker(
            TransferToObjectFIFOPattern(object_fifo_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            TransferToRuntimeSequence(object_fifo_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        # insert dma wait statements for bd collisions
        PatternRewriteWalker(ManageSyncs(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(OptimizeWaits()).rewrite_module(op)

        ## lower computation node ops for known kernels

        PatternRewriteWalker(
            ConvPattern(tile_op_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            MMPattern(tile_op_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        # handle layouts
        PatternRewriteWalker(SetKernelLayouts()).rewrite_module(op)
        PatternRewriteWalker(RealizeLayoutCats(object_fifo_manager)).rewrite_module(op)

        PatternRewriteWalker(InfinteLoopCol(), apply_recursively=False).rewrite_module(op)

        ## cleanup
        PatternRewriteWalker(EraseEdges()).rewrite_module(op)
