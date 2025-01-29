from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, StringAttr, i32
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Operation, Region, SSAValue
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
    Block,
    CoreOp,
    DeviceOp,
    EndOp,
    ObjectFifoAcquireOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    ObjectFIFOReleaseOp,
    ObjectFIFOSubviewAccessOp,
    SymbolTable,
    TileOp,
)
from xdsl_aie.dialects.aiex import (
    DmaMemcpyNdOp,
    DmaWaitOp,
    RuntimeSequenceOp,
)

from stream.compiler.dialects.stream import ComputationNodeOp, TransferOp


def get_tile(value: str) -> tuple[int, int]:
    if value == "Any":
        return (0, 0)
    elif value == "Core(0)":
        return (0, 2)
    raise ValueError("unknown tile")


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
        self.tile_ops[(x, y)] = tile_op
        return tile_op


@dataclass
class TransferToObjectFIFOPattern(RewritePattern):

    tile_op_manager: TileOpManager
    counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter):

        source_tile = self.tile_op_manager.insert_or_update(*get_tile(op.source.data))
        dest_tile = self.tile_op_manager.insert_or_update(*get_tile(op.dest.data))

        of_name = f"of{self.counter}"

        object_fifo = ObjectFifoOp(
            elemNumber=IntegerAttr(1, i32),
            producerTile=source_tile,
            consumerTiles=[dest_tile],
            referenced_type=op.results[0].type.element_type,
            shape=op.results[0].type.get_shape(),
            name=of_name,
        )

        # object fifo should be defined at start of device
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, object_fifo)

        # decide whether to consume or produce
        if op.source.data == "Any":
            port = ObjectFifoPortEnum.Consume
        else:
            port = ObjectFifoPortEnum.Produce

        assert isinstance(memref_type := op.results[0].type, MemRefType)

        acquire_op = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            object_fifo=of_name,
            shape=memref_type.get_shape(),
            element_type=memref_type.get_element_type(),
        )

        access_op = ObjectFIFOSubviewAccessOp(IntegerAttr(0, i32), acquire_op)

        release_op = ObjectFIFOReleaseOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            object_fifo=of_name,
        )

        # find use
        last_use = list(op.results[0].uses)[-1].operation
        rewriter.insert_op(release_op, InsertPoint.after(last_use))

        rewriter.replace_matched_op([acquire_op, access_op], new_results=access_op.results)

        # increment of counter
        self.counter += 1


@dataclass
class TestPatttern(RewritePattern):

    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        # four i32's?
        input_types.extend([i32] * 4)

        func_op = FuncOp(op.kernel.data, (input_types, []), Region(), "private")

        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, func_op)

        c32 = ConstantOp.from_int_and_width(32, i32)
        c64 = ConstantOp.from_int_and_width(64, i32)
        c10 = ConstantOp.from_int_and_width(10, i32)

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)
        inputs.extend([c32, c32, c64, c10])

        func_call = CallOp(op.kernel.data, inputs, [])

        rewriter.replace_matched_op((c32, c64, c10, func_call))


@dataclass
class InsertRuntimeDMAs(RewritePattern):

    sequence_op: RuntimeSequenceOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ObjectFifoOp, rewriter: PatternRewriter):

        # Add Block Argument to SequenceOp
        memref_type = op.elemType.buffer
        assert isinstance(memref_type, MemRefType)

        sequence_block = self.sequence_op.body.block

        sequence_block.insert_arg(memref_type, 0)

        # Insert DMA
        memcpy = DmaMemcpyNdOp(
            0,
            0,
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


class ConvertStreamToAIEPass(ModulePass):
    name = "convert-stream-to-aie"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # wrap everything in a device op

        rewriter = Rewriter()
        device_op = DeviceOp(
            IntegerAttr.from_int_and_width(AIEDeviceEnum.npu1_1col.get_int(), 32),
            rewriter.move_region_contents_to_new_regions(op.body),
        )
        op.body.add_block(Block([device_op]))

        # wrap everything in a core op
        core_tile = TileOp(0, 2)

        core_op = CoreOp(
            None,
            core_tile,
            rewriter.move_region_contents_to_new_regions(device_op.region),
            link_with=StringAttr("conv2dk1_i8.o"),
        )
        # add end op
        rewriter.insert_op(EndOp(), InsertPoint.at_end(core_op.region.block))
        device_op.region.add_block(Block([core_tile, core_op]))

        tile_op_manager = TileOpManager(device_op)

        PatternRewriteWalker(
            TransferToObjectFIFOPattern(tile_op_manager),
            apply_recursively=False,
        ).rewrite_module(op)
        PatternRewriteWalker(
            TestPatttern(tile_op_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        # add a runtime sequence operation
        runtime_sequence = RuntimeSequenceOp(Region(Block()))
        rewriter.insert_op(runtime_sequence, InsertPoint.at_end(device_op.region.block))

        PatternRewriteWalker(InsertRuntimeDMAs(runtime_sequence), apply_recursively=False).rewrite_module(op)
