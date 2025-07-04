from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.computation.computation_node import GeneratedComputationNode
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class ConcatConstantNode(PropagationNode):
    """Class that represents an onnx Concat node with one constant input."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        constant_shape: tuple[int, ...],
        variable_input_first: bool,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the ConcatConstantNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the input/constants are concatenated
            constant_shape: the shape of the constant tensor
            variable_input_first: Wether the result is `concat(input, constant_tensor)` or
                `concat(constant_tensor, input)`
        """
        if input_names is None:
            input_names = []
        op_type = "concat"
        super().__init__(node_id, node_name, op_type, input_names)

        self.axis = axis
        self.constant_shape = constant_shape
        self.variable_input_first = variable_input_first

        match len(predecessors):
            case 0:
                self.input_operand_source = {}
            case 1:
                self.input_operand_source = {LayerOperand("I"): predecessors[0]}
            case 2:
                # `indices` (the second input) are considered as inputs
                self.input_operand_source = {LayerOperand("W"): predecessors[0], LayerOperand("I"): predecessors[1]}
            case _:
                raise ValueError("More than two inputs for ConcatConstantNode")

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Perform gather operation on the tensor."""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        relevant_axes[self.axis] = True
        extended_tensor = tensor.concat_with_empty(
            shape=self.constant_shape, axis=self.axis, variable_input_first=self.variable_input_first
        )
        return extended_tensor, relevant_axes


class ConcatNode(PropagationNode):
    """Class that represents an onnx Concat node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        output_shape: tuple[int, ...],
        input_names: list[str] | None = None,
        axis_exists_in_input: bool = False,
    ) -> None:
        """Initialize the ConcatConstantNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the inputs are concatenated
            output_shape: the shape of the output
            axis_exists_in_input: whether the input already has the axis over which the concationation happens

        """
        if input_names is None:
            input_names = []
        op_type = "concat"
        super().__init__(node_id, node_name, op_type, input_names)
        self.axis = axis
        self.output_shape = output_shape
        self.axis_exists_in_input = axis_exists_in_input

        self.input_operand_source = {LayerOperand(f"I{i}"): node_id for i, node_id in enumerate(predecessors)}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """The input slice is only one of many inputs of this node, but the output tensor should have the shape of the
        concat node output. Return a tensor of all zeros except the input tensor at the correct index"""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        assert isinstance(previous_node, GeneratedComputationNode), (
            "Concat only supported for procedurally generated nodes for now"
        )
        assert not self.axis_exists_in_input or (
            len(tensor.tensor_shape) == len(self.output_shape) and tensor.tensor_shape[self.axis] == 1
        ), """Input tensor does not have size-1 dimension to concatenate on"""

        slice_idx = previous_node.gen_id
        extended_tensor = tensor.concat_with_empty_both_sides(
            output_shape=self.output_shape,
            axis=self.axis,
            slice_idx=slice_idx,
            axis_exists_in_input=self.axis_exists_in_input,
        )

        # Log this axis as relevant
        relevant_axes[self.axis] = True

        return extended_tensor, relevant_axes
