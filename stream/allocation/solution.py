from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stream.workload.utils import is_mac_operator_type

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core
    from stream.opt.solver import SolveStats
    from stream.workload.node import Tensor, TransferNode

#: Core kinds that model a memory/DMA endpoint, not a compute engine -- never in a compute roofline.
_NON_COMPUTE_CORE_TYPES: frozenset[str] = frozenset({"offchip", "shim", "memory"})


@dataclass(frozen=True)
class Latency:
    """The solved latencies in cycles: all iterations, one iteration, the overlap of two, and the fill."""

    total: int
    per_iteration: int
    overlap: int
    fill: int


@dataclass(frozen=True)
class AllocationSolution:
    """What the allocation solve decided for a steady state, and what it measured of it."""

    tensor_placements: Mapping[Tensor, tuple[Core, ...]]
    transfer_routes: Mapping[TransferNode, MulticastPathPlan]
    memory_cores: Mapping[TransferNode, tuple[Core, ...]]
    reuse_levels: Mapping[Tensor, int]
    depths: Mapping[Tensor, int]
    single_buffered: frozenset[Tensor]
    latency: Latency
    primary_cost: float
    throughput_bound: float
    solve_stats: SolveStats
    performance: dict[str, Any] | None
    capacity_slack: Mapping[int, dict[str, float]]


def mac_roofline_peak(accelerator: Accelerator) -> tuple[int, int]:
    """``(peak_macs_per_cycle, n_cores)`` over the on-chip cores that may execute MAC work."""
    offchip_id = accelerator.offchip_core_id
    peak = 0
    n_cores = 0
    for core in accelerator.core_list:
        if core.id == offchip_id or core.type in _NON_COMPUTE_CORE_TYPES:
            continue
        op_types = getattr(core, "operator_types", None)
        if op_types is not None and not any(is_mac_operator_type(t) for t in op_types):
            continue
        units = getattr(getattr(core, "operational_array", None), "total_unit_count", 0) or 0
        if not units:
            continue
        peak += units
        n_cores += 1
    return peak, n_cores


def end_to_end_mac_utilization(accelerator: Accelerator, total_mac_ops: int | None, latency: int) -> dict[str, Any]:
    """The aggregate stats of ``total_mac_ops / (peak_macs_per_cycle * latency)``, both restricted to the
    matmul/conv family; the utilization is None without MAC work."""
    peak, mac_cores = mac_roofline_peak(accelerator)
    util = (total_mac_ops / (peak * latency)) if (total_mac_ops and peak and latency and latency > 0) else None
    return {
        "total_mac_ops": total_mac_ops,
        "peak_macs_per_cycle": peak,
        "mac_capable_cores": mac_cores,
        "end_to_end_mac_utilization": util,
    }
