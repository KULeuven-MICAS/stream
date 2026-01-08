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
        ssis_multiplicity: int = 1,
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
        # Adjust intra_core_tiling according to ssis_multiplicity
        if self.intra_core_tiling and ssis_multiplicity > 1:
            last_dim, last_size = self.intra_core_tiling[-1]
            assert last_size % ssis_multiplicity == 0, (
                "SteadyStateComputation: The first dimension size of intra_core_tiling must be divisible by "
                f"ssis_multiplicity. Got {last_size} and {ssis_multiplicity}."
            )
            self.intra_core_tiling[-1] = (last_dim, last_size // ssis_multiplicity)
        steady_state_iteration_space = SteadyStateIterationSpace.from_computation_node(
            node=self, multiplicity=ssis_multiplicity
        )

        # Initialize SteadyStateNode (explicitly, since ComputationNode also inherits from Node)
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
