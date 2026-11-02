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
    """A fused group lowered to its steady state: ``workload`` and ``mapping`` with the placements and routes of its
    transfers, ``source_workload`` and ``source_mapping`` before lowering, and ``offchip_held`` the bits each off-chip
    memory holds of tensors already placed, beside which the inputs the allocation places must fit."""

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
    offchip_held: dict[Memory, dict[Tensor, float]] = field(default_factory=dict)
