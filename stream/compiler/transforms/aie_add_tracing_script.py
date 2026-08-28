from collections import Counter
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
    ObjectFifoOp,
    RuntimeSequenceOp,
    TileOp,
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

# What a mem tile waits on. Core event names are not mem tile events and lowering rejects
# them, so a substituted tile has to bring its own.
MEMTILE_EVENTS = (
    "DMA_MM2S_SEL0_MEMORY_STARVATION",
    "DMA_S2MM_SEL0_MEMORY_BACKPRESSURE",
    "DMA_MM2S_SEL0_STALLED_LOCK",
    "DMA_S2MM_SEL0_STALLED_LOCK",
    "DMA_MM2S_SEL0_START_TASK",
    "DMA_S2MM_SEL0_START_TASK",
)

# A trace unit takes eight event slots and mlir-aie expects all of them.
_EVENT_SLOTS = 8

# South ports on a core tile's stream switch, and the rows tracing cares about.
_SOUTH_PORTS = 4
_CORE_ROW = 2
_MEM_ROW = 1

# aie.trace.packet type for a mem tile.
_MEMTILE_PACKET = 3

# Packet ids run 1..31. Routing usually runs out first, so pick fewer than this.
MAX_TRACED_TILES = 31


def _coords(tile) -> tuple[int, int]:
    owner = tile.owner
    return (owner.col.value.data, owner.row.value.data)


def _blocked_columns(op: ModuleOp) -> set[int]:
    """Columns a core trace cannot leave.

    A core reaches the shim through the south ports of the rows under it, and the streams
    the layer already sends down that column claim those first. A stream descends in the
    column it lands in, which is not always the one it leaves.
    """
    descending: Counter[int] = Counter()
    for fifo in op.walk():
        if not isinstance(fifo, ObjectFifoOp):
            continue
        col, row = _coords(fifo.producerTile)
        if row < _CORE_ROW:
            continue
        for consumer in fifo.consumerTiles:
            consumer_col, consumer_row = _coords(consumer)
            if consumer_row < row:
                descending[consumer_col] += 1
    return {col for col, taken in descending.items() if taken >= _SOUTH_PORTS}


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
        # A column with no south port left traces its mem tile, which sits under the contention.
        blocked = _blocked_columns(op)
        mem_tiles = {
            tile.col.value.data: tile.result
            for tile in op.walk()
            if isinstance(tile, TileOp) and tile.row.value.data == _MEM_ROW
        }
        tiles = list(dict.fromkeys(mem_tiles.get(_coords(t)[0], t) if _coords(t)[0] in blocked else t for t in tiles))
        if len(tiles) > self.max_tiles:
            tiles = tiles[: self.max_tiles]

        if not tiles:
            return

        def slots(events: tuple[str, ...]) -> tuple[str, ...]:
            events = tuple(events)[:_EVENT_SLOTS]
            return events + ("NONE",) * (_EVENT_SLOTS - len(events))

        rewriter = Rewriter()
        names: list[str] = []
        for index, tile in enumerate(tiles):
            name = f"trace_core_{index}"
            names.append(name)
            on_mem_tile = _coords(tile)[1] == _MEM_ROW
            body = Block(
                [
                    TraceModeOp(),
                    TracePacketOp(_MEMTILE_PACKET if on_mem_tile else 0),
                    *(TraceEventOp(e) for e in slots(MEMTILE_EVENTS if on_mem_tile else self.events)),
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
