"""The AIE2 tile array's limits, the constraint families the ``aie2`` namespace contributes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families.dma import DMA_CHANNELS
from stream.opt.allocation.constraint_optimization.families.memory import BUFFER_DESCRIPTORS, OBJECT_FIFO_DEPTH
from stream.opt.allocation.constraint_optimization.timeslot_allocation import _resource_key

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.context import MemoryReuseEntry
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry

NAMESPACE = "aie2"


class ObjectFifoDepth:
    """An AIE2 tile's object fifos are at most its ``max_object_fifo_depth`` deep."""

    name: ClassVar[str] = "aie2_object_fifo_depth"
    requires: ClassVar[tuple[str, ...]] = ("object_fifo_depth",)
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        for core, depth in q.indexed("object_fifo_depth").items():
            if core.namespace == NAMESPACE:
                ctx.add_constr(
                    depth.expr <= core.max_object_fifo_depth,
                    name=f"aie2_obj_fifo_depth_Core_{core.id}",
                    resource=core,
                    kind=OBJECT_FIFO_DEPTH,
                    bound=float(core.max_object_fifo_depth),
                )


class BufferDescriptors:
    """An AIE2 tile's transfers use at most ``max_object_fifo_depth`` buffer descriptors."""

    name: ClassVar[str] = "aie2_buffer_descriptors"
    requires: ClassVar[tuple[str, ...]] = ("buffer_descriptor_depth",)
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        for core, depth in q.indexed("buffer_descriptor_depth").items():
            if core.namespace == NAMESPACE:
                ctx.add_constr(
                    depth.expr <= core.max_object_fifo_depth,
                    name=f"aie2_bd_depth_Core_{core.id}",
                    resource=core,
                    kind=BUFFER_DESCRIPTORS,
                    bound=float(core.max_object_fifo_depth),
                )


class MemoryReuse:
    """A memory tile outlives its reader only where one whole-object replay expresses the re-read."""

    name: ClassVar[str] = "aie2_memory_reuse"
    requires: ClassVar[tuple[str, ...]] = ("memory_reuse",)
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        entries: tuple[MemoryReuseEntry, ...] = q.get("memory_reuse").expr
        for entry in entries:
            if entry.core.namespace != NAMESPACE:
                continue
            for i, (mem_stop, compute_stop) in enumerate(entry.unexpressible):
                ctx.add_constr(
                    mem_stop._raw + compute_stop._raw <= 1,
                    name=f"aie2_mem_replay_{entry.name}_Core_{entry.core.id}_P{i}",
                    resource=entry.core,
                )


class DmaChannels:
    """A tile drives at most its DMA channels in each direction: a compute tile's, a memory tile's, or the shim's
    for the off-chip core."""

    name: ClassVar[str] = "aie2_dma_channels"
    requires: ClassVar[tuple[str, ...]] = ("dma_in", "dma_out")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        max_compute_tile_dma_channels: int = 2,
        max_mem_tile_dma_channels: int = 6,
        max_shim_tile_dma_channels: int = 2,
    ) -> None:
        self.max_compute_tile_dma_channels = max_compute_tile_dma_channels
        self.max_mem_tile_dma_channels = max_mem_tile_dma_channels
        self.max_shim_tile_dma_channels = max_shim_tile_dma_channels

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        for direction in ("in", "out"):
            for core, usage in q.indexed(f"dma_{direction}").items():
                limit = self.channels(core, ctx.space.offchip_core_id)
                ctx.add_constr(
                    usage.expr <= limit,
                    name=f"dma_{direction}_cap_{_resource_key(core)}",
                    resource=core,
                    kind=DMA_CHANNELS,
                )

    def channels(self, core: Core, offchip_core_id: int | None) -> int:
        """The DMA channels ``core`` has in each direction."""
        if core.id == offchip_core_id:
            return self.max_shim_tile_dma_channels
        if core.type == "memory":
            return self.max_mem_tile_dma_channels
        if core.type == "compute":
            return self.max_compute_tile_dma_channels
        raise ValueError(f"Unexpected core type for DMA channel constraint: {core.type}")
