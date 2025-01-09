from xdsl.context import MLContext
from xdsl.dialects.builtin import FunctionType, IntegerAttr, MemRefType, ModuleOp, StringAttr, i32
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Attribute, OpResult, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.hints import isa
from xdsl_aie.dialects.aie import (
    AIEDeviceEnum,
    Block,
    CoreOp,
    DeviceOp,
    ObjectFIFOReleaseOp,
    ObjectFifoAcquireOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    SymbolTable,
    TileOp,
)

from stream.compiler.dialects.stream import ComputationNodeOp, TransferOp


def get_tile_op(value: str) -> TileOp:
    if value == "Any":
        return TileOp(IntegerAttr(0, i32), IntegerAttr(0, i32))
    elif value == "Core(1)":
        return TileOp(IntegerAttr(0, i32), IntegerAttr(2, i32))
    raise ValueError("unknown tile")


class TransferToObjectFIFOPattern(RewritePattern):

    counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter):

        source_tile = get_tile_op(op.source.data)
        dest_tile = get_tile_op(op.dest.data)

        of_name = f"of{self.counter}"

        object_fifo = ObjectFifoOp(
            elemNumber=IntegerAttr(1, i32),
            producerTile=source_tile,
            consumerTiles=[dest_tile],
            referenced_type=op.results[0].type,
            shape=[],
            name=of_name,
        )

        # object fifo should be defined at start of device
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, object_fifo)
        rewriter.insert_op([source_tile, dest_tile], InsertPoint.before(object_fifo))

        # decide whether to consume or produce
        if op.source.data == 'Any':
            port = ObjectFifoPortEnum.Produce
        else:
            port = ObjectFifoPortEnum.Consume

        assert isinstance(memref_type := op.results[0].type, MemRefType)

        acquire_op = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            object_fifo=of_name,
            shape=memref_type.get_shape(),
            element_type=memref_type.get_element_type(),
        )

        release_op = ObjectFIFOReleaseOp(IntegerAttr.from_int_and_width(port.get_int(), 32), IntegerAttr.from_int_and_width(1, 32), object_fifo=of_name)

        # find use
        last_use = list(op.results[0].uses)[-1].operation
        rewriter.insert_op(release_op, InsertPoint.after(last_use))

        rewriter.replace_matched_op([acquire_op], new_results=acquire_op.results)

        # increment of counter
        self.counter += 1


class TestPatttern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:

        func_op = FuncOp(op.kernel.data, ([], []), Region(), "private")

        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, func_op)

        func_call = CallOp(op.kernel.data, [], [])

        rewriter.replace_matched_op(func_call)

class ConvertStreamToAIEPass(ModulePass):
    name = "convert-stream-to-aie"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:

        # wrap everything in a device op

        rewriter = Rewriter()
        device_op = DeviceOp(
            IntegerAttr.from_int_and_width(AIEDeviceEnum.npu1.get_int(), 32),
            rewriter.move_region_contents_to_new_regions(op.body),
        )
        op.body.add_block(Block([device_op]))

        # wrap everything in a core op 
        core_tile = TileOp(0, 2)

        core_op = CoreOp(
            None, 
            core_tile, 
            rewriter.move_region_contents_to_new_regions(device_op.region))
        device_op.region.add_block(Block([core_tile, core_op]))

        PatternRewriteWalker(
            GreedyRewritePatternApplier([TransferToObjectFIFOPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)


        PatternRewriteWalker(
            GreedyRewritePatternApplier([TestPatttern()]),
            apply_recursively=False,
        ).rewrite_module(op)

