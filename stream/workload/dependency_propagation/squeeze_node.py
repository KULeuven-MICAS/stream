from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class SqueezeNode(PropagationNode):
    """Class that represents an onnx unsqueeze node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        squeeze_axes: list[int] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        if input_names is None:
            input_names = []
        op_type = "unsqueeze"
        super().__init__(node_id, node_name, op_type, input_names)

        self.squeeze_axes = squeeze_axes
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Unsqueeze an input tensor"""
        squeezed_tensor = tensor.squeeze(axes=self.squeeze_axes)

        if relevant_axes is None:
            relevant_axes = [False] * len(squeezed_tensor.tensor_shape)
        if self.squeeze_axes is not None:
            for axis in self.squeeze_axes:
                del relevant_axes[axis]

        return squeezed_tensor, relevant_axes
