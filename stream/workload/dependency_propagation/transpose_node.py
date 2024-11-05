from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class TransposeNode(PropagationNode):
    """Class that represents an onnx Transpose node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        permute_axes: list[int] | None = None,
        input_names: list[str] = [],
    ) -> None:
        op_type = "transpose"
        super().__init__(node_id, node_name, op_type, input_names)

        self.permute_axes = permute_axes
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def propagate(self, tensor: NodeTensor, next_node: Node | None = None) -> NodeTensor:
        """Transpose an input tensor."""
        return tensor.transpose(axes=self.permute_axes)
