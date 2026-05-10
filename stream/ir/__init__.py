"""stream.ir — Pydantic IR models for workloads and accelerators.

Re-exports the main IR classes and their per-persona view models.
AllocationIR will be added in Plan 02.
"""

from stream.ir.accelerator import (
    AcceleratorCompilerView,
    AcceleratorHardwareView,
    AcceleratorIR,
    CoreIR,
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
]
