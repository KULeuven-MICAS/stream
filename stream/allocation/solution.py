from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.opt.solver import SolveStats
    from stream.workload.node import Tensor, TransferNode


@dataclass(frozen=True)
class Latency:
    """The solved latencies in cycles: all iterations, one iteration, the overlap of two, and the fill."""

    total: int
    per_iteration: int
    overlap: int
    fill: int


@dataclass(frozen=True)
class AllocationSolution:
    """What the allocation solve decided for a steady state, and what it measured of it: per slot its latency, per
    transfer the iterations one firing serves and the cycles its route takes, the solver's metrics, and the reports,
    each None when it could not be computed."""

    tensor_placements: Mapping[Tensor, tuple[Core, ...]]
    transfer_routes: Mapping[TransferNode, MulticastPathPlan]
    memory_cores: Mapping[TransferNode, tuple[Core, ...]]
    reuse_levels: Mapping[Tensor, int]
    depths: Mapping[Tensor, int]
    single_buffered: frozenset[Tensor]
    latency: Latency
    slot_latencies: Mapping[int, float]
    reuse_factors: Mapping[TransferNode, float]
    route_cycles: Mapping[TransferNode, int]
    primary_cost: float
    throughput_bound: float
    solve_stats: SolveStats
    metrics: dict[str, Any]
    performance: dict[str, Any] | None
    capacity_slack: Mapping[int, dict[str, float]] | None
    slot_latency_breakdown: dict[str, Any] | None
