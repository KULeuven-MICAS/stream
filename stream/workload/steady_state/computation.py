from zigzag.datatypes import LayerOperand
from zigzag.workload.layer_node import LayerNodeAttributes

from stream.hardware.architecture.core import Core
from stream.workload.computation.computation_node import ComputationNode, OperandTensorReshape
from stream.workload.mapping import InterCoreMappingAttributes
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state.node import SteadyStateNode


class SteadyStateComputation(ComputationNode, SteadyStateNode):
    """A ComputationNode that is also aware of steady-state scheduling and metrics."""

    def __init__(  # noqa: PLR0913
        self,
        id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
        mapping_attr: InterCoreMappingAttributes,
        input_names: list[str],
        possible_resource_allocation: list[Core],
        operand_tensor_reshape: OperandTensorReshape | None = None,
        produces_final_output: bool = False,
        group_id: int = 0,
        sub_id: int = -1,
        partially_constant_operands: list[LayerOperand] | None = None,
    ):
        if partially_constant_operands is None:
            partially_constant_operands = []

        # Initialize ComputationNode
        ComputationNode.__init__(
            self=self,
            node_id=id,
            node_name=node_name,
            node_attr=node_attr,
            mapping_attr=mapping_attr,
            op_type="computation",
            operand_tensor_reshape=operand_tensor_reshape,
            produces_final_output=produces_final_output,
            group_id=group_id,
            sub_id=sub_id,
            input_names=input_names,
            partially_constant_operands=partially_constant_operands,
        )

        # For now, assume the steady state iteration space is not important for the computation nodes
        steady_state_iteration_space = SteadyStateIterationSpace([])

        # Initialize SteadyStateNode (explicitly, since ComputationNode also inherits from Node)
        possible_resource_allocation = []
        SteadyStateNode.__init__(
            self=self,
            id=id,
            node_name=node_name,
            type="computation",
            possible_resource_allocation=possible_resource_allocation,
            steady_state_iteration_space=steady_state_iteration_space,
        )
        self.chosen_resource_allocation = possible_resource_allocation[0] if possible_resource_allocation else None

    @property
    def plot_name(self):
        return self.node_name
