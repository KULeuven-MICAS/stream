from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.config import (
    ConstraintOptStageConfig,
    CoreConstraintProfile,
    CoreRole,
    ensure_profile_registry,
    pick_profile,
)
from stream.opt.allocation.constraint_optimization.timeslot_allocation import _resource_key

if TYPE_CHECKING:
    import gurobipy as gp

logger = logging.getLogger(__name__)


# ============================================================================
# ConstraintContext – used by the *timeslot* allocation stage
# ============================================================================


@dataclass(frozen=True)
class ConstraintContext:
    core_profiles: dict[Core, CoreConstraintProfile]
    compute_cores: list[Core]
    cache_cores: list[Core]
    io_cores: list[Core]
    capacities: dict[Core, float]
    offchip_core_id: int | None

    def core_ids(self) -> list[int]:
        return sorted(c.id for c in self.compute_cores)


def build_constraint_context(accelerator: Accelerator, cfg: ConstraintOptStageConfig) -> ConstraintContext:
    profiles_registry = ensure_profile_registry(cfg.profiles)

    core_profiles: dict[Core, CoreConstraintProfile] = {}
    for core in accelerator.core_list:
        core_profiles[core] = pick_profile(profiles_registry, core.type)

    def _cores_with(role: CoreRole) -> list[Core]:
        return [c for c, p in core_profiles.items() if role in p.roles]

    def _eligible_for_compute(core: Core) -> bool:
        profile = core_profiles[core]
        return bool(profile.roles)

    compute_cores = [
        c
        for c in core_profiles
        if _eligible_for_compute(c) and (accelerator.offchip_core_id is None or c.id != accelerator.offchip_core_id)
    ]
    cache_cores = _cores_with(CoreRole.CACHE)
    io_cores = _cores_with(CoreRole.IO) + _cores_with(CoreRole.SHIM)

    capacities: dict[Core, float] = {c: c.get_memory_capacity() for c in compute_cores}

    return ConstraintContext(
        core_profiles=core_profiles,
        compute_cores=compute_cores,
        cache_cores=cache_cores,
        io_cores=io_cores,
        capacities=capacities,
        offchip_core_id=accelerator.offchip_core_id,
    )


# ============================================================================
# Namespace-specific MILP constraint strategies
# ----------------------------------------------------------------------------
# Every core namespace (e.g. "aie2", "zigzag") may impose additional
# hardware-specific constraints on the transfer / tensor allocation MILP.
#
# HOW TO ADD A NEW NAMESPACE
# --------------------------
#   1. Subclass NamespaceConstraints below.
#   2. Override any of the add_*_constraints methods you need.
#   3. In build_transfer_context(), detect when the namespace is present
#      in the accelerator and instantiate your strategy with appropriate
#      parameters.
# ============================================================================


class NamespaceConstraints:
    """Base class for namespace-specific MILP constraints.

    Subclasses set :attr:`NAMESPACE` and override the ``add_*_constraints``
    methods they need.  Methods that are *not* overridden default to a no-op,
    so only the constraints relevant to a namespace are ever emitted.
    """

    NAMESPACE: str = ""

    def applies_to(self, core: Core) -> bool:
        """Return ``True`` if *core* belongs to this namespace."""
        return core.namespace == self.NAMESPACE

    # ---- object-FIFO depth ----

    def add_object_fifo_constraints(
        self,
        model: gp.Model,
        object_fifo_depth: dict[Core, gp.LinExpr],
    ) -> None:
        """Enforce FIFO-depth limits for cores in this namespace."""

    # ---- buffer descriptors ----

    def add_buffer_descriptor_constraints(
        self,
        model: gp.Model,
        buffer_descriptor_depth: dict[Core, gp.LinExpr],
    ) -> None:
        """Enforce buffer-descriptor limits for cores in this namespace."""

    # ---- DMA channel usage ----

    def add_dma_usage_constraints(
        self,
        model: gp.Model,
        dma_usage_in: dict[Core, gp.Var],
        dma_usage_out: dict[Core, gp.Var],
    ) -> list[gp.Var]:
        """Enforce DMA channel limits.

        Returns a (possibly empty) list of Gurobi variables representing
        penalty terms that the caller should include in the MILP objective.
        """
        return []


class AIE2Constraints(NamespaceConstraints):
    """Hardware constraints specific to the AIE2 tile array.

    * Object-FIFO depth: each tile has a per-core ``max_object_fifo_depth``.
    * DMA channels: the mem-tile and shim-tile have a finite number of S2MM /
      MM2S DMA channels.  The peak usage across all tiles of each kind is
      constrained to the respective hardware limit.
    """

    NAMESPACE = "aie2"

    def __init__(
        self,
        *,
        offchip_core_id: int | None,
        max_compute_tile_dma_channels: int = 8,
        max_mem_tile_dma_channels: int = 6,
        max_shim_tile_dma_channels: int = 2,
    ) -> None:
        self.offchip_core_id = offchip_core_id
        self.max_compute_tile_dma_channels = max_compute_tile_dma_channels
        self.max_mem_tile_dma_channels = max_mem_tile_dma_channels
        self.max_shim_tile_dma_channels = max_shim_tile_dma_channels

    # ---- object-FIFO depth ----

    def add_object_fifo_constraints(
        self,
        model: gp.Model,
        object_fifo_depth: dict[Core, gp.LinExpr],
    ) -> None:
        for core, expr in object_fifo_depth.items():
            if not self.applies_to(core):
                continue
            model.addConstr(
                expr <= core.max_object_fifo_depth,
                name=f"aie2_obj_fifo_depth_Core_{core.id}",
            )

    # ---- buffer descriptors ----

    def add_buffer_descriptor_constraints(
        self,
        model: gp.Model,
        buffer_descriptor_depth: dict[Core, gp.LinExpr],
    ) -> None:
        for core, expr in buffer_descriptor_depth.items():
            if not self.applies_to(core):
                continue
            model.addConstr(
                expr <= core.max_object_fifo_depth,
                name=f"aie2_bd_depth_Core_{core.id}",
            )

    # ---- DMA channel usage ----
    def get_max_dma_channels(self, core: Core) -> int:
        if core.id == self.offchip_core_id:
            return self.max_shim_tile_dma_channels
        elif core.type == "memory":
            return self.max_mem_tile_dma_channels
        elif core.type == "compute":
            return self.max_compute_tile_dma_channels
        else:
            raise ValueError(f"Unexpected core type for DMA channel constraint: {core.type}")

    def add_dma_usage_constraints(
        self,
        model: gp.Model,
        dma_usage_in: dict[Core, gp.Var],
        dma_usage_out: dict[Core, gp.Var],
    ) -> list[gp.Var]:
        # Filter to aie2 cores only
        for core, v_in in dma_usage_in.items():
            max_in = self.get_max_dma_channels(core)
            model.addConstr(v_in <= max_in, name=f"dma_in_cap_{_resource_key(core)}")

        for core, v_out in dma_usage_out.items():
            max_out = self.get_max_dma_channels(core)
            model.addConstr(v_out <= max_out, name=f"dma_out_cap_{_resource_key(core)}")


# ============================================================================
# TransferAndTensorContext – used by the *transfer / tensor* allocation stage
# ============================================================================


@dataclass(frozen=True)
class TransferAndTensorContext:
    """Shared context for the transfer and tensor allocation MILP.

    Contains universal topology information and a list of
    :class:`NamespaceConstraints` strategies that add hardware-specific
    constraints to the model.
    """

    offchip_core_id: int | None
    mem_cores: list[Core]
    force_double_buffering: bool
    force_io_transfers_on_mem_tile: bool
    namespace_constraints: tuple[NamespaceConstraints, ...] = ()

    # ---- dispatch helpers ----

    def add_object_fifo_constraints(
        self,
        model: gp.Model,
        object_fifo_depth: dict[Core, gp.LinExpr],
    ) -> None:
        """Dispatch object-FIFO depth constraints to all namespace strategies."""
        for ns in self.namespace_constraints:
            ns.add_object_fifo_constraints(model, object_fifo_depth)

    def add_buffer_descriptor_constraints(
        self,
        model: gp.Model,
        buffer_descriptor_depth: dict[Core, gp.LinExpr],
    ) -> None:
        """Dispatch buffer-descriptor constraints to all namespace strategies."""
        for ns in self.namespace_constraints:
            ns.add_buffer_descriptor_constraints(model, buffer_descriptor_depth)

    def add_dma_usage_constraints(
        self,
        model: gp.Model,
        dma_usage_in: dict[Core, gp.Var],
        dma_usage_out: dict[Core, gp.Var],
    ) -> None:
        """Dispatch DMA-usage constraints to all namespace strategies.

        Returns the union of objective-penalty variables from every strategy.
        """
        for ns in self.namespace_constraints:
            ns.add_dma_usage_constraints(
                model,
                dma_usage_in,
                dma_usage_out,
            )


def build_transfer_context(
    accelerator: Accelerator,
    *,
    nb_cols_to_use: int = 4,
    force_double_buffering: bool = True,
    force_io_transfers_on_mem_tile: bool = True,
    max_compute_tile_dma_channels: int = 8,
    max_mem_tile_dma_channels: int = 6,
    max_shim_tile_dma_channels: int = 2,
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

    # Detect which namespaces are present and instantiate their constraint
    # strategies.  Each namespace appears at most once.
    namespaces: set[str] = {c.namespace for c in accelerator.core_list if isinstance(c, Core) and c.namespace}

    ns_constraints: list[NamespaceConstraints] = []
    if "aie2" in namespaces:
        ns_constraints.append(
            AIE2Constraints(
                offchip_core_id=offchip_core_id,
                max_compute_tile_dma_channels=max_compute_tile_dma_channels,
                max_mem_tile_dma_channels=max_mem_tile_dma_channels,
                max_shim_tile_dma_channels=max_shim_tile_dma_channels,
            )
        )
    # Future namespaces: add elif / append blocks here.

    return TransferAndTensorContext(
        offchip_core_id=offchip_core_id,
        mem_cores=mem_cores,
        force_double_buffering=force_double_buffering,
        force_io_transfers_on_mem_tile=force_io_transfers_on_mem_tile,
        namespace_constraints=tuple(ns_constraints),
    )
