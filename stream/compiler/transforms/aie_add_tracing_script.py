from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import (
    CoreOp,
    DeviceOp,
    EndOp,
    RuntimeSequenceOp,
    TraceEventOp,
    TraceHostConfigOp,
    TraceModeOp,
    TraceOp,
    TracePacketOp,
    TraceStartConfigOp,
    TraceStartOp,
    TraceStopOp,
)

# The instruction events bracket each kernel call and the stall events explain the gaps.
# No port events: those need a DMA channel, which is only assigned later.
DEFAULT_EVENTS = (
    "INSTR_EVENT_0",
    "INSTR_EVENT_1",
    "MEMORY_STALL",
    "LOCK_STALL",
    "INSTR_VECTOR",
)

# A trace unit takes eight event slots and mlir-aie expects all of them.
_EVENT_SLOTS = 8

# Packet ids run 1..31. Routing usually runs out first, so pick fewer than this.
MAX_TRACED_TILES = 31


def _coords(tile) -> tuple[int, int]:
    owner = tile.owner
    return (owner.col.value.data, owner.row.value.data)


@dataclass(frozen=True)
class AIEAddTracingScript(ModulePass):
    """Emit trace configuration for the tiles that run kernels.

    Lowering turns it into the packet flow, shim allocation and register writes, and
    appends a trace buffer to the runtime sequence for the host to supply.
    """

    name = "aie-add-tracing-script"

    trace_size: int = 1048576
    max_tiles: int = MAX_TRACED_TILES
    events: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVENTS)
    # (col, row) to trace. Empty means every tile with a kernel, up to max_tiles.
    tiles: tuple[tuple[int, int], ...] = ()

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        device = next((o for o in op.walk() if isinstance(o, DeviceOp)), None)
        sequence = next((o for o in op.walk() if isinstance(o, RuntimeSequenceOp)), None)
        if device is None or sequence is None:
            return

        # Tracing a tile only says something if a kernel runs on it.
        tiles = list(dict.fromkeys(core.tile for core in op.walk() if isinstance(core, CoreOp)))
        if self.tiles:
            wanted = set(self.tiles)
            tiles = [t for t in tiles if _coords(t) in wanted]
            missing = wanted - {_coords(t) for t in tiles}
            if missing:
                raise ValueError(f"no kernel runs on tile(s) {sorted(missing)}, so they cannot be traced")
        if len(tiles) > self.max_tiles:
            tiles = tiles[: self.max_tiles]

        if not tiles:
            return

        events = tuple(self.events)[:_EVENT_SLOTS]
        events += ("NONE",) * (_EVENT_SLOTS - len(events))

        rewriter = Rewriter()
        names: list[str] = []
        for index, tile in enumerate(tiles):
            name = f"trace_core_{index}"
            names.append(name)
            body = Block(
                [
                    TraceModeOp(),
                    TracePacketOp(),
                    *(TraceEventOp(event) for event in events),
                    TraceStartOp(),
                    TraceStopOp(),
                    EndOp(),
                ]
            )
            # The device block already ends in aie.end.
            rewriter.insert_op(
                TraceOp(name, tile, Region(body)),
                InsertPoint.before(device.region.block.last_op),
            )

        configs = [TraceHostConfigOp(self.trace_size), *(TraceStartConfigOp(n) for n in names)]
        rewriter.insert_op(configs, InsertPoint.at_start(sequence.body.block))
