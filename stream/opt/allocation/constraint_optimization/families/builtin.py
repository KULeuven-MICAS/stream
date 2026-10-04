"""Stream's own constraint families, each one of the allocator's constraint groups."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE
from stream.opt.solver import PipeliningModel

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )


class Placement:
    """Each movable tensor takes exactly one of its placements."""

    name: ClassVar[str] = "placement"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._tensor_placement_constraints()


class PathChoice:
    """Each transfer takes one route, whose ends hold the tensors it moves."""

    name: ClassVar[str] = "path_choice"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._path_choice_constraints()


class ReuseRates:
    """The reuse factor of each transfer: how many iterations one firing serves, from its tensor's reuse stop."""

    name: ClassVar[str] = "reuse_rates"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("reuse_factor",)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._reuse_factor_rate_constraints()


class LinkContention:
    """A link carries at most one transfer per slot."""

    name: ClassVar[str] = "link_contention"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._link_contention_constraints()


class MemoryCapacity:
    """What each memory holds fits in its capacity, less what the toolchain reserves."""

    name: ClassVar[str] = "memory_capacity"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._memory_capacity_constraints()


class ObjectFifoDepth:
    """The object-fifo depth each core's tensors need, per core; a namespace family bounds it."""

    name: ClassVar[str] = "object_fifo_depth"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("object_fifo_depth",)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._object_fifo_depth_constraints()
        for core, depth in alloc.object_fifo_depth.items():
            q.add("object_fifo_depth", depth, index=core)


class BufferDescriptors:
    """The buffer descriptors each core's transfers need, per core; a namespace family bounds them."""

    name: ClassVar[str] = "buffer_descriptors"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("buffer_descriptor_depth",)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._buffer_descriptor_constraints()
        for core, depth in alloc.bd_depth.items():
            q.add("buffer_descriptor_depth", depth, index=core)


class SlotLatency:
    """A slot lasts as long as the slowest node or transfer in it; the longest any can take bounds every slot."""

    name: ClassVar[str] = "slot_latency"
    requires: ClassVar[tuple[str, ...]] = ("reuse_factor",)
    provides: ClassVar[tuple[str, ...]] = ("transfer_latency", SLOT_PRESSURE)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._slot_latency_constraints()
        q.add(SLOT_PRESSURE, 0, index="slot_latency", upper_bound=alloc._longest_step())


class ReuseLevels:
    """A tensor handed from core to core is held up to its outermost irrelevant loop."""

    name: ClassVar[str] = "reuse_levels"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._force_nonconstant_reuse_levels()


class OutputReuse:
    """A final output is held up to its outermost irrelevant loop."""

    name: ClassVar[str] = "output_reuse"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._force_final_output_reuse_levels()


class ReuseCompatibility:
    """The reuse levels on either side of a transfer between a memory and a compute tile agree; the residency a
    memory tile keeps beyond its reader is what a namespace family checks it can replay."""

    name: ClassVar[str] = "reuse_compatibility"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("memory_reuse",)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        q.add("memory_reuse", tuple(alloc._ensure_memory_and_compute_reuse_compatibility()))


class SpatialReuse:
    """Reuse covers every temporal loop inside a tensor's outermost spatial loop."""

    name: ClassVar[str] = "spatial_reuse"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._force_reuse_includes_spatial()


class Overlap:
    """How much of an iteration the next one overlaps, and the fill before the first. ``model`` is ``occupancy``
    (any slot a resource leaves unused) or ``span`` (only before its first and after its last use); with
    ``transfer_contention`` and ``offchip_contention`` a busy on-chip or off-chip link bounds the overlap."""

    name: ClassVar[str] = "overlap"
    requires: ClassVar[tuple[str, ...]] = ("transfer_latency", "reuse_factor", SLOT_PRESSURE)
    provides: ClassVar[tuple[str, ...]] = ("overlap", "iteration", "fill", "shared_busy")

    def __init__(
        self,
        model: str | PipeliningModel = PipeliningModel.OCCUPANCY,
        transfer_contention: bool = True,
        offchip_contention: bool = True,
    ) -> None:
        self.model = PipeliningModel(model)
        self.transfer_contention = transfer_contention
        self.offchip_contention = offchip_contention

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._overlap(self.model, self.transfer_contention, self.offchip_contention)


class DmaChannels:
    """The DMA channels each core's transfers drive in and out, whose peaks the latency objective charges; a
    namespace family bounds them."""

    name: ClassVar[str] = "dma_channels"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("dma_in", "dma_out", "dma_peak_in", "dma_peak_out")

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        alloc._add_dma_usage_constraints()
        for core, usage in alloc.core_dma_in.items():
            q.add("dma_in", usage, index=core)
        for core, usage in alloc.core_dma_out.items():
            q.add("dma_out", usage, index=core)
        q.add("dma_peak_in", alloc.max_core_dma_in._raw)
        q.add("dma_peak_out", alloc.max_core_dma_out._raw)


class OffchipTraffic:
    """The bits crossing the off-chip boundary, charged in the latency objective at the time the off-chip links
    take to move them; nothing is charged where a shared-bandwidth model already times them."""

    name: ClassVar[str] = "offchip_traffic"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("offchip_traffic_weight",)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        if not alloc.shared_bandwidth and (bandwidth := alloc._offchip_bandwidth()):
            q.add("offchip_traffic_weight", alloc.iterations / bandwidth)
