from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.core import Core
from stream.workload.workload import ComputationNode


class CorePerformanceEstimator(Protocol):
    """Protocol for core performance estimators."""

    def estimate(self, node: ComputationNode, core: Core, core_id: int) -> CoreCostEntry: ...


@dataclass
class ZigZagPerformanceEstimator:
    run_zigzag: Any
    increase_cc_per_op: Any
    check_core_capacity_for_node: Any
    cost_lut: Any
    copy_fn: Any

    def estimate(self, node: ComputationNode, core: Core, core_id: int) -> CoreCostEntry:
        node_duplicate = self.copy_fn(node)
        self.cost_lut.remove_cores_with_same_id(node, core)
        too_large_operands_for_cme = self.check_core_capacity_for_node(core, node_duplicate)
        node_duplicate.set_chosen_core_allocation(core_id)
        if core.dataflows:
            node_duplicate.spatial_mapping = core.dataflows
        cme = self.run_zigzag(node_duplicate, too_large_operands_for_cme, core_id)
        cme = self.increase_cc_per_op(cme, node.type)
        node_duplicate.set_chosen_core_allocation(None)
        return CoreCostEntry(
            energy_total=getattr(cme, "energy_total", 0),
            latency_total=getattr(cme, "latency_total2", getattr(cme, "ideal_cycle", 0)),
            ideal_cycle=getattr(cme, "ideal_cycle", 0),
            ideal_temporal_cycle=getattr(cme, "ideal_temporal_cycle", 0),
            mem_energy_breakdown=getattr(cme, "mem_energy_breakdown", {}),
            cme=cme,
            mapping=getattr(cme, "mapping", None),
            layer=node,
        )


@dataclass
class AIEPerformanceEstimator:
    """Simple utilization-based estimator for AIE compute cores."""

    def estimate(self, node: ComputationNode, core: Core, core_id: int) -> CoreCostEntry:
        macs = getattr(node, "total_mac_count", 0) or 0
        utilization = getattr(node.kernel, "utilization", 100.0) or 100.0
        cycles = macs * (100.0 / utilization)
        energy = macs * (100.0 / utilization)
        return CoreCostEntry(
            energy_total=energy,
            latency_total=cycles,
            ideal_cycle=cycles,
            ideal_temporal_cycle=cycles,
            mem_energy_breakdown={},
            cme=None,
            mapping=None,
            layer=node,
            metadata={"utilization": utilization, "core_id": core_id},
        )
