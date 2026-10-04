from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES
from stream.plugins import load_group

if TYPE_CHECKING:
    from stream.opt.solver import LinExpr, SolverVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryReuseEntry:
    """One staged tensor's residency on a memory tile, held against its reader's."""

    name: str
    core: Core
    mem_level: LinExpr
    compute_level: LinExpr
    unexpressible: tuple[tuple[SolverVar, SolverVar], ...]


# ============================================================================
# Namespace facts and constraint families
# ----------------------------------------------------------------------------
# Every core namespace (e.g. "aie2", "zigzag") may state facts the allocation
# model reads (which cores share memory, what the toolchain reserves, what a
# dispatch costs) and contribute constraint families to the default set.
#
# HOW TO ADD A NEW NAMESPACE
# --------------------------
#   1. Subclass NamespaceConstraints, set NAMESPACE, override the facts that
#      differ, list the namespace's constraint families in ``families``, and
#      override from_config if the subclass takes constructor arguments.
#   2. Register it in the "stream.constraints" entry-point group under the
#      namespace name, and its families in "stream.constraint_families". They
#      are then picked up whenever the accelerator has a core of that
#      namespace -- in-tree here, or from an overlay with no fork.
# ============================================================================


@dataclass(frozen=True)
class NamespaceConstraintConfig:
    """What a namespace's constraints may be built from."""

    accelerator: Accelerator
    offchip_core_id: int | None
    mem_cores: tuple[Core, ...]
    nb_cols_to_use: int


class NamespaceConstraints:
    """What a namespace tells the allocation model about its cores, and the constraint families it adds to
    the default set. Subclasses set :attr:`NAMESPACE` and override the facts that differ from the defaults."""

    NAMESPACE: str = ""
    families: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: NamespaceConstraintConfig) -> NamespaceConstraints:
        """Build this strategy for one solve. Override when the subclass needs constructor arguments."""
        return cls()

    def applies_to(self, core: Core) -> bool:
        """Return ``True`` if *core* belongs to this namespace."""
        return core.namespace == self.NAMESPACE

    def shares_memory(self, one: Core, other: Core) -> bool:
        """Whether one core reaches the other's memory without spending a DMA channel.

        Namespaces that have no such path keep the default, which charges every transfer
        a channel.
        """
        return False

    def reserved_memory_bits(self, core: Core) -> int:
        """Bits of the core's data memory the toolchain claims before any tensor lands."""
        return 0

    def dispatch_overhead_cycles(self, columns_per_design: Sequence[int]) -> float:
        """Cycles one dispatch spends configuring, given each design's column span."""
        return 0.0


class AIE2Constraints(NamespaceConstraints):
    """The AIE2 tile array: neighbouring tiles share memory, a core's stack is reserved, a dispatch of several
    designs reconfigures their columns, and the object-fifo, buffer-descriptor, memory-tile replay and DMA
    channel limits are its families."""

    NAMESPACE = "aie2"
    families = ("aie2_object_fifo_depth", "aie2_buffer_descriptors", "aie2_memory_reuse", "aie2_dma_channels")

    def __init__(self, *, reconfiguration: Mapping[str, float] | None = None) -> None:
        reconfiguration = reconfiguration or {}
        self.cycles_per_column = float(reconfiguration.get("cycles_per_column", 0.0))
        self.reset_cycles = float(reconfiguration.get("reset_cycles", 0.0))

    @classmethod
    def from_config(cls, config: NamespaceConstraintConfig) -> AIE2Constraints:
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
        """Whether two tiles are served by one memory module.

        On AIE2 a core reaches the memory north and south of it, the memory west of it and
        its own -- ``isMemEast`` is ``isInternal``, so there is no east neighbour. The test
        is still symmetric because a fifo only needs one of its two ends to reach the
        other, which is what ``isSharedMemory`` decides by trying both orders.

        ``isLegalMemAffinity`` also excludes a memory tile to the south, which the core
        type already rules out here.
        """
        if not (self.applies_to(one) and self.applies_to(other)):
            return False
        if one.type != "compute" or other.type != "compute":
            return False
        if None in (one.col_id, one.row_id, other.col_id, other.row_id):
            return False
        return abs(one.col_id - other.col_id) + abs(one.row_id - other.row_id) == 1


# ============================================================================
# TransferAndTensorContext – used by the *transfer / tensor* allocation stage
# ============================================================================


@dataclass(frozen=True)
class TransferAndTensorContext:
    """Shared context for the transfer and tensor allocation MILP: the topology, and the
    :class:`NamespaceConstraints` of the namespaces the accelerator has cores of."""

    accelerator: Accelerator
    offchip_core_id: int | None
    mem_cores: list[Core]
    force_double_buffering: bool
    force_io_transfers_on_mem_tile: bool
    namespace_constraints: tuple[NamespaceConstraints, ...] = ()

    @property
    def default_families(self) -> tuple[str, ...]:
        """Stream's own constraint families and those each namespace contributes."""
        return (*DEFAULT_FAMILIES, *(family for ns in self.namespace_constraints for family in ns.families))

    def shares_memory(self, one: Core, other: Core) -> bool:
        """Whether the two cores use one memory, or a namespace lets one reach the other's."""
        if self.accelerator.memory_of(one) == self.accelerator.memory_of(other):
            return True
        return any(ns.shares_memory(one, other) for ns in self.namespace_constraints)

    def reserved_memory_bits(self, core: Core) -> int:
        """Bits the toolchain claims on this core, summed over the namespaces that own it."""
        return sum(ns.reserved_memory_bits(core) for ns in self.namespace_constraints if ns.applies_to(core))

    def dispatch_overhead_cycles(self, columns_per_design: Sequence[int]) -> float:
        """Configuration cycles one dispatch pays, summed over the namespaces."""
        return sum(ns.dispatch_overhead_cycles(columns_per_design) for ns in self.namespace_constraints)


CONSTRAINTS_GROUP = "stream.constraints"


def namespace_constraints_for(
    accelerator: Accelerator, config: NamespaceConstraintConfig
) -> list[NamespaceConstraints]:
    """The constraint strategies for the namespaces this accelerator contains (from the
    ``stream.constraints`` entry-point group, keyed by namespace name)."""
    present = {c.namespace for c in accelerator.core_list if isinstance(c, Core) and c.namespace}
    strategies: dict[str, NamespaceConstraints] = {}
    for plugin in load_group(CONSTRAINTS_GROUP):
        if plugin.name not in present:
            continue
        try:
            strategies[plugin.name] = plugin.obj.from_config(config)
        except Exception as exc:  # noqa: BLE001 -- a broken strategy must not fail the solve
            logger.warning("skipping %r constraints from %r: %s", plugin.name, plugin.distribution, exc)
    for namespace in sorted(present - strategies.keys()):
        logger.debug("no MILP constraints registered for core namespace %r", namespace)
    return [strategies[name] for name in sorted(strategies)]


def build_transfer_context(
    accelerator: Accelerator,
    *,
    nb_cols_to_use: int = 4,
    force_double_buffering: bool = True,
    force_io_transfers_on_mem_tile: bool = True,
) -> TransferAndTensorContext:
    offchip_core_id = accelerator.offchip_core_id

    # Memory cores eligible for on-chip caching (not off-chip, memory kind,
    # with known coordinates inside the column budget).
    mem_cores: list[Core] = [
        c
        for c in accelerator.core_list
        if isinstance(c, Core)
        and c.id != offchip_core_id
        and c.kind == "memory"
        and c.col_id is not None
        and c.col_id < nb_cols_to_use
    ]

    config = NamespaceConstraintConfig(
        accelerator=accelerator,
        offchip_core_id=offchip_core_id,
        mem_cores=tuple(mem_cores),
        nb_cols_to_use=nb_cols_to_use,
    )
    ns_constraints = tuple(namespace_constraints_for(accelerator, config))

    return TransferAndTensorContext(
        accelerator=accelerator,
        offchip_core_id=offchip_core_id,
        mem_cores=mem_cores,
        force_double_buffering=force_double_buffering,
        force_io_transfers_on_mem_tile=force_io_transfers_on_mem_tile,
        namespace_constraints=ns_constraints,
    )
