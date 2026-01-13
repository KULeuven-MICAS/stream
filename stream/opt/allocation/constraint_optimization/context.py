from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gurobipy import GRB

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.config import (
    ConstraintOptStageConfig,
    CoreConstraintProfile,
    CoreRole,
    ensure_profile_registry,
    pick_profile,
)

if TYPE_CHECKING:
    import gurobipy as gp


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


def build_constraint_context(
    accelerator: Accelerator, cfg: ConstraintOptStageConfig, compute_cfg_override: ComputeMilpConfig | None = None
) -> ConstraintContext:
    profiles_registry = ensure_profile_registry(cfg.profiles)
    compute_cfg = compute_cfg_override or cfg.compute

    core_profiles: dict[Core, CoreConstraintProfile] = {}
    for core in accelerator.core_list:
        core_profiles[core] = pick_profile(profiles_registry, core.type)

    def _cores_with(role: CoreRole) -> list[Core]:
        return [c for c, p in core_profiles.items() if role in p.roles]

    def _eligible_for_compute(core: Core) -> bool:
        profile = core_profiles[core]
        return bool(profile.roles & compute_cfg.eligible_roles)

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


@dataclass(frozen=True)
class TransferAndTensorContext:
    offchip_core_id: int | None
    mem_cores: list[Core]
    force_double_buffering: bool
    force_io_transfers_on_mem_tile: bool
    max_mem_tile_dma_channels: int
    max_shim_tile_dma_channels: int
    object_fifo_cores: set[Core]

    def add_object_fifo_constraints(self, model: gp.Model, object_fifo_depth: dict[Core, gp.LinExpr]) -> None:
        for core, expr in object_fifo_depth.items():
            if core not in self.object_fifo_cores:
                continue
            model.addConstr(expr <= core.max_object_fifo_depth, name=f"obj_fifo_depth_Core {core.id}")

    def add_dma_usage_constraints(
        self,
        model,
        mem_core_usage_s2mm: dict[Core, gp.Var],
        mem_core_usage_mm2s: dict[Core, gp.Var],
        shim_core_usage_s2mm: dict[Core, gp.Var],
        shim_core_usage_mm2s: dict[Core, gp.Var],
    ) -> tuple[gp.Var, gp.Var]:
        max_mem_core_usage = model.addVar(vtype=GRB.INTEGER, name="maxMemCoreUsage")
        for i, usage in enumerate(mem_core_usage_s2mm.values()):
            model.addConstr(max_mem_core_usage >= usage, name=f"maxMemCoreUsageS2MM_le_{i}")
        for i, usage in enumerate(mem_core_usage_mm2s.values()):
            model.addConstr(max_mem_core_usage >= usage, name=f"maxMemCoreUsageMM2S_le_{i}")
        model.addConstr(
            max_mem_core_usage <= self.max_mem_tile_dma_channels,
            name=f"maxMemCoreUsage_le_{self.max_mem_tile_dma_channels}",
        )

        max_shim_core_usage = model.addVar(vtype=GRB.INTEGER, name="maxShimCoreUsage")
        for i, usage in enumerate(shim_core_usage_s2mm.values()):
            model.addConstr(max_shim_core_usage >= usage, name=f"maxShimCoreUsageS2MM_le_{i}")
        for i, usage in enumerate(shim_core_usage_mm2s.values()):
            model.addConstr(max_shim_core_usage >= usage, name=f"maxShimCoreUsageMM2S_le_{i}")
        model.addConstr(
            max_shim_core_usage <= self.max_shim_tile_dma_channels,
            name=f"maxShimCoreUsage_le_{self.max_shim_tile_dma_channels}",
        )

        return max_mem_core_usage, max_shim_core_usage


def build_transfer_context(
    accelerator: Accelerator,
    *,
    nb_cols_to_use: int = 4,
    force_double_buffering: bool = True,
    force_io_transfers_on_mem_tile: bool = True,
    max_mem_tile_dma_channels: int = 6,
    max_shim_tile_dma_channels: int = 2,
) -> TransferAndTensorContext:
    offchip_core_id = accelerator.offchip_core_id
    mem_cores: list[Core] = [
        c
        for c in accelerator.core_list
        if isinstance(c, Core)
        and c.id != offchip_core_id
        and c.type == "memory"
        and c.col_id is not None
        and c.col_id < nb_cols_to_use
    ]
    object_fifo_cores = {
        c
        for c in accelerator.core_list
        if isinstance(c, Core)
        and c.type in {"compute", "memory"}
        and isinstance(getattr(c, "core_type", ""), str)
        and c.core_type.startswith("aie")
    }
    return TransferAndTensorContext(
        offchip_core_id=offchip_core_id,
        mem_cores=mem_cores,
        force_double_buffering=force_double_buffering,
        force_io_transfers_on_mem_tile=force_io_transfers_on_mem_tile,
        max_mem_tile_dma_channels=max_mem_tile_dma_channels,
        max_shim_tile_dma_channels=max_shim_tile_dma_channels,
        object_fifo_cores=object_fifo_cores,
    )
