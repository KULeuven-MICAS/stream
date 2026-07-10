"""stream.ir — Pydantic IR models for workloads, accelerators, and allocations.

Re-exports the main IR classes and their per-persona view models.
"""

from stream.ir.accelerator import (
    AcceleratorCompilerView,
    AcceleratorHardwareView,
    AcceleratorIR,
    CoreIR,
)
from stream.ir.allocation import (
    AllocationAlgorithmicView,
    AllocationCompilerView,
    AllocationHardwareView,
    AllocationIR,
    AllocationPerformanceView,
    BottleneckIR,
    ConstraintSelectionIR,
    CostModelsIR,
    FusedGroupIR,
    LatencyInfo,
    NodeAllocationIR,
    NodePerformanceIR,
    PerformanceAggregateIR,
)
from stream.ir.graph_view import (
    BlockClassIR,
    GraphEdgeIR,
    GraphNodeIR,
    RegionIR,
    WorkloadGraphView,
)
from stream.ir.workload import (
    NodeIR,
    WorkloadAlgorithmicView,
    WorkloadCompilerView,
    WorkloadIR,
)

__all__ = [
    "WorkloadIR",
    "WorkloadAlgorithmicView",
    "WorkloadCompilerView",
    "NodeIR",
    "WorkloadGraphView",
    "GraphNodeIR",
    "GraphEdgeIR",
    "BlockClassIR",
    "RegionIR",
    "AcceleratorIR",
    "AcceleratorHardwareView",
    "AcceleratorCompilerView",
    "CoreIR",
    "AllocationIR",
    "AllocationAlgorithmicView",
    "AllocationHardwareView",
    "AllocationCompilerView",
    "AllocationPerformanceView",
    "CostModelsIR",
    "BottleneckIR",
    "PerformanceAggregateIR",
    "NodePerformanceIR",
    "LatencyInfo",
    "ConstraintSelectionIR",
    "NodeAllocationIR",
    "FusedGroupIR",
]
