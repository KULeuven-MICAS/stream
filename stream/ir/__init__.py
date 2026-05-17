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
    ConstraintSelectionIR,
    FusedGroupIR,
    LatencyInfo,
    NodeAllocationIR,
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
    "LatencyInfo",
    "ConstraintSelectionIR",
    "NodeAllocationIR",
    "FusedGroupIR",
]
