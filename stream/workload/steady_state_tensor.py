from enum import Flag, auto

from zigzag.datatypes import LayerOperand

from stream.hardware.architecture.core import Core
from stream.workload.steady_state_iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state_node import SteadyStateNode


class TensorFlag(Flag):
    CONSTANT = auto()
    NONCONSTANT = auto()
    INPUT = auto()
    OUTPUT = auto()


class SteadyStateTensor(SteadyStateNode):
    """Abstract base class for nodes representing tensors in the graph."""

    def __init__(
        self,
        id: int,
        node_name: str,
        size: int,
        type: TensorFlag,  # Replace with actual TensorFlag enum
        operand: LayerOperand,
        steady_state_iteration_space: SteadyStateIterationSpace,
        possible_resource_allocation: list[Core | None] = [],
    ):
        super().__init__(
            id=id,
            node_name=node_name,
            type="tensor",
            steady_state_iteration_space=steady_state_iteration_space,
            possible_resource_allocation=possible_resource_allocation,
        )
        self.size = size
        self.tensor_flag = type
        self.operand = operand
        self.runtime = 0

    def __str__(self):
        return f"Tensor({self.node_name})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.id, self.node_name, self.tensor_flag, self.chosen_resource_allocation))

    def __eq__(self, other):
        if not isinstance(other, SteadyStateTensor):
            return False
        return (
            self.id == other.id
            and self.node_name == other.node_name
            and self.tensor_flag == other.tensor_flag
            and self.chosen_resource_allocation == other.chosen_resource_allocation
        )

    @property
    def plot_name(self):
        return self.node_name
