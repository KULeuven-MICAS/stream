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
    FusedGroupIR,
    LatencyInfo,
    NodeAllocationIR,
    NodePerformanceIR,
    PerformanceAggregateIR,
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
    "AcceleratorIR",
    "AcceleratorHardwareView",
    "AcceleratorCompilerView",
    "CoreIR",
    "AllocationIR",
    "AllocationAlgorithmicView",
    "AllocationHardwareView",
    "AllocationCompilerView",
    "AllocationPerformanceView",
    "BottleneckIR",
    "PerformanceAggregateIR",
    "NodePerformanceIR",
    "LatencyInfo",
    "ConstraintSelectionIR",
    "NodeAllocationIR",
    "FusedGroupIR",
]
