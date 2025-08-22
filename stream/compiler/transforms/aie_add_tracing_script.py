import aie.utils.trace as trace_utils
from aie.dialects.aie import AIEDevice, device, tile
from aie.extras.context import mlir_mod_ctx
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import DeviceOp, TileOp
from xdsl_aie.dialects.aiex import RuntimeSequenceOp


class AIEAddTracingScript(ModulePass):
    name = "aie-add-tracing-script"

    def __init__(self, trace_size=1048576):
        self.trace_size = trace_size

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        rewriter = Rewriter()

        # 1: Get packet flow
        # find shim tile and compute tile:
        shim_tile = None
        # compute_tile2 = None
        compute_tiles: dict[tuple[int, int], TileOp] = {}
        for tile_op in op.walk():
            if isinstance(tile_op, TileOp):
                match tile_idx := (tile_op.col.value.data, tile_op.row.value.data):
                    case (0, 0):
                        shim_tile = tile_op
                    case _:
                        compute_tiles[tile_idx] = tile_op
        assert shim_tile is not None

        with mlir_mod_ctx() as aie_ctx:

            @device(AIEDevice.npu2)
            def device_body():
                shim_tile_aie = tile(0, 0)
                tiles_to_trace = []
                for tile_idx in compute_tiles:
                    tile_aie = tile(*tile_idx)
                    tiles_to_trace.append(tile_aie)
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile_aie)

        packet_flow_str = aie_ctx.module.body.operations[0].get_asm(print_generic_op_form=True)

        # modify to fit into our own IR:
        parser = Parser(Context(allow_unregistered=True), packet_flow_str)
        module = parser.parse_module()

        packet_flow_ops: list[Operation] = [
            op for op in module.body.block.first_op.regions[0].block.ops if op.op_name.data == "aie.packet_flow"
        ]
        for packet_flow_op, compute_tile in zip(packet_flow_ops, compute_tiles.values(), strict=True):
            packet_flow_op.detach()
            packet_flow_op.regions[0].block.first_op.operands[0] = compute_tile.result
            packet_flow_op.regions[0].block.first_op.next_op.operands[0] = shim_tile.result
        # for op in module.body.block.first_op.regions[0].block.ops:
        #     if op.op_name.data in ("aie.tile", "aie.end"):
        #         continue
        #     op.detach()
        #     op.regions[0].block.first_op.operands[0] = compute_tile2.result
        #     op.regions[0].block.first_op.next_op.operands[0] = shim_tile.result
        #     packet_flow_ops.append(op)
        # packet_flow_op = module.body.block.first_op.regions[0].block.last_op.prev_op  # pyright: ignore
        # assert isinstance(packet_flow_op, Operation)
        # packet_flow_op.detach()
        # packet_flow_op.regions[0].block.first_op.operands[0] = compute_tile2.result
        # packet_flow_op.regions[0].block.first_op.next_op.operands[0] = shim_tile.result

        # insert into device body:
        for device_op in op.walk():
            if isinstance(device_op, DeviceOp):
                rewriter.insert_op(packet_flow_ops, InsertPoint.at_end(device_op.region.block))

        # 2: Runtime sequence thingies
        #
        with mlir_mod_ctx() as aie_ctx:

            @device(AIEDevice.npu2)
            def device_body():
                shim_tile_aie = tile(0, 0)
                tiles_to_trace = []
                for tile_idx in compute_tiles:
                    tile_aie = tile(*tile_idx)
                    tiles_to_trace.append(tile_aie)

                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=shim_tile_aie,
                    trace_size=self.trace_size,
                    coretile_events=[
                        # captures input A (PORT_RUNNING_0, at port number 1, master for inputs)
                        trace_utils.PortEvent(
                            trace_utils.CoreEvent.PORT_RUNNING_0,
                            port_number=1,
                            master=True,
                        ),
                        # captures input B (PORT_RUNNING_1, at port number 2, master for inputs)
                        trace_utils.PortEvent(
                            trace_utils.CoreEvent.PORT_RUNNING_1,
                            port_number=2,
                            master=True,
                        ),
                        # captures output C (PORT_RUNNING_2, at port number 1, slave for outputs)
                        trace_utils.PortEvent(
                            trace_utils.CoreEvent.PORT_RUNNING_2,
                            port_number=1,
                            master=False,
                        ),
                        trace_utils.CoreEvent.INSTR_EVENT_0,
                        trace_utils.CoreEvent.INSTR_EVENT_1,
                        trace_utils.CoreEvent.MEMORY_STALL,
                        trace_utils.CoreEvent.LOCK_STALL,
                        trace_utils.CoreEvent.INSTR_VECTOR,
                    ],
                )

        runtime_str = aie_ctx.module.body.operations[0].get_asm(print_generic_op_form=True)

        parser = Parser(Context(allow_unregistered=True), runtime_str)
        module = parser.parse_module()

        # Find runtime sequence:
        for runtime_sequence in op.walk():
            if isinstance(runtime_sequence, RuntimeSequenceOp):
                # Insert all ops at start
                insert_point = InsertPoint.at_start(runtime_sequence.body.block)
                for operation in module.body.block.first_op.regions[0].block.ops:
                    if operation.op_name.data in ("aie.tile", "aie.end"):
                        continue
                    operation.detach()
                    rewriter.insert_op(operation, insert_point)
                    insert_point = InsertPoint.after(operation)
