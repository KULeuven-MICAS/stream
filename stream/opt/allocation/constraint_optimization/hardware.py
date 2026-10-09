"""What the allocation model knows of the hardware: the topology, and the facts each core namespace states."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.plugins import load_group

logger = logging.getLogger(__name__)

NAMESPACES_GROUP = "stream.namespaces"
REMOVED_GROUP = "stream.constraints"

REMOVED_HOOKS = {
    "add_object_fifo_constraints": "aie2_object_fifo_depth, which bounds the object_fifo_depth quantity",
    "add_memory_reuse_constraints": "aie2_memory_reuse, which reads the memory_reuse quantity",
    "add_buffer_descriptor_constraints": "aie2_buffer_descriptors, which bounds the buffer_descriptor_depth quantity",
    "add_dma_usage_constraints": "aie2_dma_channels, which bounds the dma_in and dma_out quantities",
}
"""The constraint hooks a namespace had before 2.0, each with the family that replaces it in the ``aie2`` namespace."""


@dataclass(frozen=True)
class NamespaceConfig:
    """What a namespace's facts may be built from."""

    accelerator: Accelerator


class HardwareNamespace:
    """What a core namespace tells the allocation model about its cores, and the constraint families it adds to the
    default set. A subclass sets :attr:`NAMESPACE`, overrides the facts that differ and lists its ``families``; it is
    registered in the ``stream.namespaces`` entry-point group under the namespace name."""

    NAMESPACE: str = ""
    families: tuple[str, ...] = ()
    pairs_by_spatial_index: bool = False
    """Whether its code generator matches the target at ``j`` with the sources at ``j``, ``j + m``, ... (``m`` the
    narrower side) whatever the tilings place where, instead of each side holding its tiles in core order."""
    accumulates_across_cores: bool = True
    """Whether a computation split over a dimension it reduces can be completed by adding its cores' partial sums."""
    circuit_switched: bool = False
    """Whether a transfer holds every link it crosses to itself for its slot, instead of sharing their bandwidth."""

    @classmethod
    def from_config(cls, config: NamespaceConfig) -> HardwareNamespace:
        """Build this namespace for one solve. Override when the subclass needs constructor arguments."""
        return cls()

    def applies_to(self, core: Core) -> bool:
        """Return ``True`` if *core* belongs to this namespace."""
        return core.namespace == self.NAMESPACE

    def shares_memory(self, one: Core, other: Core) -> bool:
        """Whether one core reaches the other's memory without spending a DMA channel; by default no transfer
        does, so every transfer is charged a channel."""
        return False

    def reserved_memory_bits(self, core: Core) -> int:
        """Bits of the core's data memory the toolchain claims before any tensor lands."""
        return 0

    def dispatch_overhead_cycles(self, columns_per_design: Sequence[int]) -> float:
        """Cycles one dispatch spends configuring, given each design's column span."""
        return 0.0


class AIE2Namespace(HardwareNamespace):
    """The AIE2 tile array: neighbouring tiles share memory, a core's stack is reserved, a dispatch of several
    designs reconfigures their columns, and the object-fifo, buffer-descriptor, memory-tile replay and DMA
    channel limits are its families."""

    NAMESPACE = "aie2"
    families = ("aie2_object_fifo_depth", "aie2_buffer_descriptors", "aie2_memory_reuse", "aie2_dma_channels")
    pairs_by_spatial_index = True
    accumulates_across_cores = False
    circuit_switched = True

    def __init__(self, *, reconfiguration: Mapping[str, float] | None = None) -> None:
        reconfiguration = reconfiguration or {}
        self.cycles_per_column = float(reconfiguration.get("cycles_per_column", 0.0))
        self.reset_cycles = float(reconfiguration.get("reset_cycles", 0.0))

    @classmethod
    def from_config(cls, config: NamespaceConfig) -> AIE2Namespace:
        return cls(reconfiguration=config.accelerator.reconfiguration)

    def dispatch_overhead_cycles(self, columns_per_design: Sequence[int]) -> float:
        """A dispatch of several designs configures each one's columns and resets the array once."""
        if len(columns_per_design) <= 1:
            return 0.0
        return sum(c * self.cycles_per_column for c in columns_per_design) + self.reset_cycles

    DEFAULT_CORE_STACK_BYTES = 1024

    def reserved_memory_bits(self, core: Core) -> int:
        if core.type != "compute":
            return 0
        return self.DEFAULT_CORE_STACK_BYTES * 8

    def shares_memory(self, one: Core, other: Core) -> bool:
        """Whether two compute tiles are served by one memory module: on AIE2 a core reaches the memory north, south
        and west of it and its own, and ``isSharedMemory`` tries both ends of a fifo, so the test is symmetric."""
        if not (self.applies_to(one) and self.applies_to(other)):
            return False
        if one.type != "compute" or other.type != "compute":
            return False
        if None in (one.col_id, one.row_id, other.col_id, other.row_id):
            return False
        return abs(one.col_id - other.col_id) + abs(one.row_id - other.row_id) == 1


BUILTIN_NAMESPACES: dict[str, type[HardwareNamespace]] = {"aie2": AIE2Namespace}


@dataclass(frozen=True)
class HardwareFacts:
    """The topology the allocation model is built on, and the :class:`HardwareNamespace` of each namespace the
    accelerator has cores of."""

    accelerator: Accelerator
    offchip_core_id: int | None
    mem_cores: list[Core]
    namespaces: tuple[HardwareNamespace, ...] = ()

    def shares_memory(self, one: Core, other: Core) -> bool:
        """Whether the two cores use one memory, or a namespace lets one reach the other's."""
        if self.accelerator.memory_of(one) == self.accelerator.memory_of(other):
            return True
        return any(ns.shares_memory(one, other) for ns in self.namespaces)

    def pairs_by_spatial_index(self, cores: Sequence[Core]) -> bool:
        """Whether a namespace of these cores pairs a transfer's sources and targets by spatial index."""
        return any(ns.pairs_by_spatial_index and ns.applies_to(core) for ns in self.namespaces for core in cores)

    def circuit_switched(self, cores: Sequence[Core]) -> bool:
        """Whether a namespace of these cores gives a transfer each link it crosses to itself for its slot."""
        return any(ns.circuit_switched and ns.applies_to(core) for ns in self.namespaces for core in cores)

    def reserved_memory_bits(self, core: Core) -> int:
        """Bits the toolchain claims on this core, summed over the namespaces that own it."""
        return sum(ns.reserved_memory_bits(core) for ns in self.namespaces if ns.applies_to(core))

    def dispatch_overhead_cycles(self, columns_per_design: Sequence[int]) -> float:
        """Configuration cycles one dispatch pays, summed over the namespaces."""
        return sum(ns.dispatch_overhead_cycles(columns_per_design) for ns in self.namespaces)


def namespace_classes(accelerator: Accelerator) -> dict[str, type[HardwareNamespace]]:
    """By name, the namespace of each core namespace ``accelerator`` has: Stream's own, overridden by those the
    ``stream.namespaces`` entry points register; one that defines a hook removed in 2.0 raises."""
    _warn_removed_group()
    present = {c.namespace for c in accelerator.core_list if isinstance(c, Core) and c.namespace}
    classes = {name: cls for name, cls in BUILTIN_NAMESPACES.items() if name in present}
    classes |= {plugin.name: plugin.obj for plugin in load_group(NAMESPACES_GROUP) if plugin.name in present}
    for name, cls in classes.items():
        if hooks := [hook for hook in REMOVED_HOOKS if hasattr(cls, hook)]:
            raise TypeError(
                f"Namespace {name!r} defines {', '.join(hooks)}, which Stream 2.0 no longer calls: a namespace "
                f"adds constraint families instead, named in its `families`, such as "
                f"{'; '.join(REMOVED_HOOKS[hook] for hook in hooks)}"
            )
    for namespace in sorted(present - classes.keys()):
        logger.debug("no allocation facts registered for core namespace %r", namespace)
    return dict(sorted(classes.items()))


@cache
def _warn_removed_group() -> None:
    for plugin in load_group(REMOVED_GROUP):
        logger.warning(
            "%r registers namespace %r in the %r entry-point group, which Stream 2.0 ignores; register it in %r",
            plugin.distribution,
            plugin.name,
            REMOVED_GROUP,
            NAMESPACES_GROUP,
        )


def namespaces_for(accelerator: Accelerator, config: NamespaceConfig) -> list[HardwareNamespace]:
    """The namespaces of :func:`namespace_classes`, built for one solve; one that cannot be built is skipped."""
    built: list[HardwareNamespace] = []
    for name, cls in namespace_classes(accelerator).items():
        try:
            built.append(cls.from_config(config))
        except Exception as exc:  # noqa: BLE001 -- a broken namespace must not fail the solve
            logger.warning("skipping the %r namespace: %s", name, exc)
    return built


def build_hardware_facts(accelerator: Accelerator, nb_cols_to_use: int) -> HardwareFacts:
    """The facts of ``accelerator`` for a solve on its first ``nb_cols_to_use`` columns; the memory cores it may
    cache on are its on-chip memory cores inside that budget."""
    offchip_core_id = accelerator.offchip_core_id
    mem_cores: list[Core] = [
        c
        for c in accelerator.core_list
        if isinstance(c, Core)
        and c.id != offchip_core_id
        and c.kind == "memory"
        and c.col_id is not None
        and c.col_id < nb_cols_to_use
    ]
    return HardwareFacts(
        accelerator=accelerator,
        offchip_core_id=offchip_core_id,
        mem_cores=mem_cores,
        namespaces=tuple(namespaces_for(accelerator, NamespaceConfig(accelerator))),
    )
