import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, StringAttr, i32
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Attribute, Operation, Region, SSAValue
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

from stream.compiler.dialects.stream import ComputationNodeOp, EdgeOp, TransferOp


def get_tile(value: str) -> tuple[int, int]:
    match = re.match(r"Core\((\d+)\)", value)
    if match:
        return 0, int(match.group(1))
    else:
        raise ValueError(f"Invalid tile value: {value}")


def get_of_name(source: TileOp, dest: TileOp, operand: str) -> str:
    of_name: str = "of_"
    of_name += f"{source.col.value.data}{source.row.value.data}to"
    of_name += f"{dest.col.value.data}{dest.row.value.data}_"
    of_name += operand
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
        self.tile_ops[(x, y)] = tile_op
        return tile_op


@dataclass
class ObjectFifoManager:

    tile_op_manager: TileOpManager
    sequence_op: RuntimeSequenceOp
    device_op: DeviceOp

    def insert_or_update(self, transfer: TransferOp) -> ObjectFifoOp:

        source_tile = self.tile_op_manager.insert_or_update(*get_tile(transfer.source.data))
        dest_tile = self.tile_op_manager.insert_or_update(*get_tile(transfer.dest.data))

        assert isinstance(memref_type := transfer.results[0].type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref_type)

        # this will reuse objectfifos of the same source dest, and type.
        of_name = get_of_name(source_tile, dest_tile, transfer.tensor.data[-2])

        object_fifo = ObjectFifoOp(
            elemNumber=IntegerAttr(1, i32),
            producerTile=source_tile,
            consumerTiles=[dest_tile],
            referenced_type=memref_type.get_element_type(),
            shape=memref_type.get_shape(),
            name=of_name,
        )

        # object fifo should be defined at start of device
        replaced = SymbolTable.insert_or_update(self.device_op, object_fifo)

        # for now, don't let this add runtime sequence ops, this needs to be done by
        # the transfer transform itself

        return object_fifo

    def of_from_name(self, name: str) -> ObjectFifoOp:
        result = SymbolTable.lookup_symbol(self.device_op, name)
        assert isinstance(result, ObjectFifoOp)
        return result

    def update_depths(self):

        current_fifo_depth: dict[str, int] = defaultdict(lambda: 0)

        for op in self.device_op.region.block.walk():
            if isinstance(op, ObjectFifoAcquireOp):
                of_name = op.objFifo_name.root_reference.data
                current_fifo_depth[of_name] += 1
                of = self.of_from_name(of_name)
                if of.elemNumber.value.data < current_fifo_depth[of_name]:
                    of.elemNumber = IntegerAttr.from_int_and_width(current_fifo_depth[of_name], 32)
            elif isinstance(op, ObjectFIFOReleaseOp):
                current_fifo_depth[op.objFifo_name.root_reference.data] -= 1


@dataclass
class TransferToObjectFIFOPattern(RewritePattern):

    object_fifo_manager: ObjectFifoManager

    release_op: dict[str, Operation | None] = field(default_factory=dict)  # pyright: ignore

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter):

        of = self.object_fifo_manager.insert_or_update(op)
        of_name = of.sym_name.data

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

        # find first and last use in this region
        assert op.parent
        operation_uses = set(x.operation for x in op.results[0].uses)
        first_use_op: Operation = next(o for o in op.parent.walk() if o in operation_uses)
        while op.parent_op() is not first_use_op.parent_op():
            assert (parent := first_use_op.parent_op()) is not None
            first_use_op = parent

        # last use OR release op of previous acquire. depending on which comes first
        if of_name not in self.release_op:
            self.release_op[of_name] = None
        last_use_op: Operation = next(
            o for o in op.parent.walk(reverse=True) if o in operation_uses or o is self.release_op[of_name]
        )
        while op.parent_op() is not last_use_op.parent_op():
            assert (parent := last_use_op.parent_op()) is not None
            last_use_op = parent

        self.release_op[of_name] = last_use_op

        rewriter.insert_op(release_op, InsertPoint.after(last_use_op))
        rewriter.insert_op([acquire_op, access_op], InsertPoint.before(first_use_op))

        op.results[0].replace_by(access_op.results[0])
        rewriter.erase_matched_op()

        # insert runtime sequence memcpy
        runtime_sequence = self.object_fifo_manager.sequence_op

        arg_order = ["I", "W", "O"]
        arg_index = arg_order.index(op.tensor.data[-2])
        arg = runtime_sequence.body.block.args[arg_index]

        static_offsets = cast(tuple[int], op.offsets.get_values())
        static_sizes = cast(tuple[int], op.sizes.get_values())
        static_strides = cast(tuple[int], op.strides.get_values())

        static_offsets = (0,) * (4 - len(static_offsets)) + static_offsets
        static_sizes = (1,) * (4 - len(static_sizes)) + static_sizes
        static_strides = (0,) * (4 - len(static_strides)) + static_strides
        static_strides = (0, 0, 64, 1)

        ids = {"I": 0, "W": 1, "O": 2}

        # Insert DMA
        memcpy = DmaMemcpyNdOp(
            0,
            0,
            arg,
            static_offsets=static_offsets,
            static_sizes=static_sizes,
            static_strides=static_strides,
            metadata=of_name,
            id=ids[op.tensor.data[-2]],
            issue_token=True,
        )

        rewriter.insert_op(memcpy, InsertPoint.at_end(runtime_sequence.body.block))


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

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)

        func_call = CallOp(op.kernel.data, inputs, [])

        rewriter.replace_matched_op(func_call)


@dataclass
class ConvPattern(RewritePattern):

    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:

        if op.kernel.data != "conv2d_k1_i8":
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


@dataclass
class EraseEdges(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EdgeOp, rewriter: PatternRewriter) -> None:
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
        )
        # add end op
        rewriter.insert_op(EndOp(), InsertPoint.at_end(core_op.region.block))
        device_op.region.add_block(Block([core_tile, core_op]))

        # add a runtime sequence operation
        # find all edges
        edges: list[EdgeOp] = [edge for edge in op.walk() if isinstance(edge, EdgeOp)]
        order = ["I", "W", "O"]

        runtime_arg_types = []
        for operand_name in order:
            edge = next(edge for edge in edges if edge.tensor.data[-2] == operand_name)
            runtime_arg_types.append(edge.output.type)

        runtime_sequence = RuntimeSequenceOp(Region(Block(arg_types=runtime_arg_types)))
        rewriter.insert_op(runtime_sequence, InsertPoint.at_end(device_op.region.block))

        tile_op_manager = TileOpManager(device_op)
        object_fifo_manager = ObjectFifoManager(tile_op_manager, runtime_sequence, device_op)

        PatternRewriteWalker(
            TransferToObjectFIFOPattern(object_fifo_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        object_fifo_manager.update_depths()

        ## lower computation node ops for known kernels

        PatternRewriteWalker(
            ConvPattern(tile_op_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            MMPattern(tile_op_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        # insert dma wait statements for bd collisions

        PatternRewriteWalker(ManageSyncs(), apply_recursively=False).rewrite_module(op)

        ## cleanup

        PatternRewriteWalker(EraseEdges()).rewrite_module(op)
