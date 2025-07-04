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
        input_names: list[str] | None = None,
    ) -> None:
        if input_names is None:
            input_names = []
        op_type = "transpose"
        super().__init__(node_id, node_name, op_type, input_names)

        self.permute_axes = permute_axes
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        """Transpose an input tensor."""
        transposed_tensor = tensor.transpose(axes=self.permute_axes)
        if self.permute_axes is not None:
            for axis in self.permute_axes:
                relevant_axes[axis] = True

        return transposed_tensor, relevant_axes
