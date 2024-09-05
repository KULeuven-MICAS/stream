from zigzag.datatypes import LayerOperand
from zigzag.workload.LayerNodeABC import LayerNodeABC

from stream.utils import NodeTensor
from stream.workload.node import Node


class ConcatNode(Node, LayerNodeABC):
    """Class that represents an onnx Concat node with one constant input."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        axis: int,
        constant_shape: tuple[int, ...],
        variable_input_first: bool,
    ) -> None:
        """Initialize the ConcatNode

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which the input/constants are concatenated
            constant_shape: the shape of the constant tensor
            variable_input_first: Wether the result is `concat(input, constant_tensor)` or
                `concat(constant_tensor, input)`
        """
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type="gather",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

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

    def concat(self, tensor: NodeTensor) -> NodeTensor:
        """Perform gather operation on the tensor."""
        return tensor.concat_with_empty(
            shape=self.constant_shape, axis=self.axis, variable_input_first=self.variable_input_first
        )
