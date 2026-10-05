"""The AIE2 tile array's limits, the constraint families the ``aie2`` namespace contributes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families.dma import DMA_CHANNELS
from stream.opt.allocation.constraint_optimization.families.memory import BUFFER_DESCRIPTORS, OBJECT_FIFO_DEPTH
from stream.opt.allocation.constraint_optimization.hardware import AIE2Namespace
from stream.opt.allocation.constraint_optimization.utils import resource_key

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind
    from stream.opt.allocation.constraint_optimization.families.reuse import MemoryReuseEntry
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext


class _DepthLimit:
    """Each AIE2 tile's ``quantity`` stays within its ``max_object_fifo_depth``, a constraint of limit ``kind``."""

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]]
    provides: ClassVar[tuple[str, ...]] = ()
    quantity: ClassVar[str]
    kind: ClassVar[ResourceKind]
    prefix: ClassVar[str]

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        for core, depth in (q.indexed(self.quantity) if self.quantity in q else {}).items():
            if core.namespace == AIE2Namespace.NAMESPACE:
                ctx.add_constr(
                    depth.expr <= core.max_object_fifo_depth,
                    name=f"{self.prefix}_Core_{core.id}",
                    resource=core,
                    kind=self.kind,
                    bound=float(core.max_object_fifo_depth),
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
            if entry.core.namespace != AIE2Namespace.NAMESPACE:
                continue
            for i, (mem_stop, compute_stop) in enumerate(entry.unexpressible):
                ctx.add_constr(
                    mem_stop._raw + compute_stop._raw <= 1,
                    name=f"aie2_mem_replay_{entry.name}_Core_{entry.core.id}_P{i}",
                    resource=entry.core,
                )


class AIE2DmaChannels:
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

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        for direction in ("in", "out"):
            for core, usage in q.indexed(f"dma_{direction}").items():
                limit = self.channels(core, ctx.space.offchip_core_id)
                ctx.add_constr(
                    usage.expr <= limit,
                    name=f"dma_{direction}_cap_{resource_key(core)}",
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
