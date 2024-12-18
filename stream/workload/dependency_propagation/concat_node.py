from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class ConcatNode(PropagationNode):
    """Class that represents an onnx Concat node with one constant input."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        constant_shape: tuple[int, ...],
        variable_input_first: bool,
        input_names: list[str] = [],
    ) -> None:
        """Initialize the ConcatNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the input/constants are concatenated
            constant_shape: the shape of the constant tensor
            variable_input_first: Wether the result is `concat(input, constant_tensor)` or
                `concat(constant_tensor, input)`
        """
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
                raise ValueError("More than two inputs for ConcatNode")

    def propagate(self, tensor: NodeTensor, next_node: Node | None = None) -> NodeTensor:
        """Perform gather operation on the tensor."""
        return tensor.concat_with_empty(
            shape=self.constant_shape, axis=self.axis, variable_input_first=self.variable_input_first
        )
