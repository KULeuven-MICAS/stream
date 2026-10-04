from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.allocation.schedule import IterationSpaces
    from stream.cost_model.core_cost_lut import CoreCostLUT
    from stream.datatypes import LayerDim
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.mapping.mapping import Mapping
    from stream.opt.allocation.constraint_optimization.context import TransferAndTensorContext
    from stream.workload.node import Node
    from stream.workload.workload import Workload


@dataclass(frozen=True)
class SteadyStateProblem:
    """A fused group lowered to its steady state, ready to allocate: ``workload`` holds its transfers, whose
    mapping lists the placements and routes each may take; ``source_workload`` is the group before lowering."""

    source_workload: Workload
    workload: Workload
    mapping: Mapping
    fusion_splits: dict[LayerDim, int]
    cost_lut: CoreCostLUT
    ssis: IterationSpaces
    iterations: int
    timeslots: dict[Node, int]
    accelerator: Accelerator
    transfer_context: TransferAndTensorContext
