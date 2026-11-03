from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.allocation.allocation import IterationSpaces
    from stream.cost_model.core_cost_lut import CoreCostLUT
    from stream.datatypes import LayerDim
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.memory import Memory
    from stream.mapping.mapping import Mapping
    from stream.opt.allocation.constraint_optimization.hardware import HardwareFacts
    from stream.workload.node import Node, Tensor
    from stream.workload.workload import Workload


@dataclass(frozen=True)
class AllocationProblem:
    """A fused group lowered to its steady state: ``workload`` and ``mapping`` with its transfers' placements and
    routes, ``source_*`` before lowering, ``held`` the bits each memory holds of placed tensors, beside which an
    off-chip memory fits the inputs the allocation places and an on-chip one the group's tensors."""

    source_workload: Workload
    source_mapping: Mapping
    workload: Workload
    mapping: Mapping
    fusion_splits: dict[LayerDim, int]
    cost_lut: CoreCostLUT
    ssis: IterationSpaces
    iterations: int
    timeslots: dict[Node, int]
    accelerator: Accelerator
    hardware: HardwareFacts
    held: dict[Memory, dict[Tensor, float]] = field(default_factory=dict)
