"""The AIE2 tile array's limits, the constraint families the ``aie2`` namespace contributes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families.dma import DMA_CHANNELS
from stream.opt.allocation.constraint_optimization.families.memory import BUFFER_DESCRIPTORS, OBJECT_FIFO_DEPTH
from stream.opt.allocation.constraint_optimization.hardware import AIE2Namespace
from stream.opt.allocation.constraint_optimization.utils import resource_key

if TYPE_CHECKING:
    from stream.hardware.architecture.memory import Memory
    from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind
    from stream.opt.allocation.constraint_optimization.families.reuse import MemoryReuseEntry
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext


class _DepthLimit:
    """Each AIE2 tile's memory's ``quantity`` stays within the tile's ``max_object_fifo_depth``, a constraint of
    limit ``kind``."""

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]]
    provides: ClassVar[tuple[str, ...]] = ()
    quantity: ClassVar[str]
    kind: ClassVar[ResourceKind]
    prefix: ClassVar[str]

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        for memory, depth in (q.indexed(self.quantity) if self.quantity in q else {}).items():
            if memory.namespace == AIE2Namespace.NAMESPACE:
                limit = memory.core.max_object_fifo_depth
                ctx.add_constr(
                    depth.expr <= limit,
                    name=f"{self.prefix}_{resource_key(memory)}",
                    resource=memory,
                    kind=self.kind,
                    bound=float(limit),
                )


class AIE2ObjectFifoDepth(_DepthLimit):
    """An AIE2 tile's object fifos are at most its ``max_object_fifo_depth`` deep."""

    name = "aie2_object_fifo_depth"
    requires = ("object_fifo_depth",)
    quantity = "object_fifo_depth"
    kind = OBJECT_FIFO_DEPTH
    prefix = "aie2_obj_fifo_depth"


class AIE2BufferDescriptors(_DepthLimit):
    """An AIE2 tile's transfers use at most ``max_object_fifo_depth`` buffer descriptors."""

    name = "aie2_buffer_descriptors"
    requires = ("buffer_descriptor_depth",)
    quantity = "buffer_descriptor_depth"
    kind = BUFFER_DESCRIPTORS
    prefix = "aie2_bd_depth"


class AIE2MemoryReuse:
    """A memory tile outlives its reader only where one whole-object replay expresses the re-read."""

    name: ClassVar[str] = "aie2_memory_reuse"
    requires: ClassVar[tuple[str, ...]] = ("memory_reuse",)
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        entries: tuple[MemoryReuseEntry, ...] = ctx.quantities.get("memory_reuse").expr
        for entry in entries:
            if entry.memory.namespace != AIE2Namespace.NAMESPACE:
                continue
            for i, (mem_stop, compute_stop) in enumerate(entry.unexpressible):
                ctx.add_constr(
                    mem_stop._raw + compute_stop._raw <= 1,
                    name=f"aie2_mem_replay_{entry.name}_{resource_key(entry.memory)}_P{i}",
                    resource=entry.memory,
                )


class AIE2DmaChannels:
    """A memory drives at most its DMA's channels in each direction: a tile's memory the tile DMA's, as its core
    declares them (S2MM channels write into the tile, MM2S channels read out of it), else the two a compute tile and
    the six a memory tile have. DRAM is reached through the shim DMAs its ports are, one per column, so each port
    drives at most the channels it declares, counted by the routes passing it; an off-chip memory whose ports declare
    none is unbounded."""

    name: ClassVar[str] = "aie2_dma_channels"
    requires: ClassVar[tuple[str, ...]] = ("dma_in", "dma_out")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        max_compute_tile_dma_channels: int = 2,
        max_mem_tile_dma_channels: int = 6,
    ) -> None:
        self.max_compute_tile_dma_channels = max_compute_tile_dma_channels
        self.max_mem_tile_dma_channels = max_mem_tile_dma_channels

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        for direction in ("in", "out"):
            for memory, usage in q.indexed(f"dma_{direction}").items():  # tile memories; DRAM's are per port below
                if (limit := self.channels(memory, direction)) is None:
                    continue
                ctx.add_constr(
                    usage.expr <= limit,
                    name=f"dma_{direction}_cap_{resource_key(memory)}",
                    resource=memory,
                    kind=DMA_CHANNELS,
                    bound=float(limit),
                )
            name = f"dma_port_{direction}"
            for (memory, port_name), usage in (q.indexed(name) if name in q else {}).items():
                channels = memory.ports[port_name].channels
                ctx.add_constr(
                    usage.expr <= channels,
                    name=f"dma_{direction}_cap_{resource_key(memory)}_{port_name}",
                    resource=memory,
                    kind=DMA_CHANNELS,
                    bound=float(channels),
                )

    def channels(self, memory: Memory, direction: str) -> int | None:
        """The DMA channels a tile's ``memory`` has for fifos ``direction`` (``in``: writing into it, ``out``: reading
        out of it); None where nothing bounds them. The off-chip memory's are its ports', each bounded apart by
        :meth:`build`."""
        incoming = direction == "in"
        if memory.offchip:
            raise ValueError(f"{memory} is off-chip: its channels are its DMA ports', bounded per port")
        # Each viewing tile's DMA, as its core declares it (an AIE2 tile's memory has one); none declared, the kind's
        if declared := sum(getattr(core, "dma_s2mm" if incoming else "dma_mm2s", 0) for core in memory.cores):
            return declared
        if memory.kind == "memory":
            return self.max_mem_tile_dma_channels
        if memory.kind == "compute":
            return self.max_compute_tile_dma_channels
        raise ValueError(f"Unexpected memory kind for DMA channel constraint: {memory.kind}")
