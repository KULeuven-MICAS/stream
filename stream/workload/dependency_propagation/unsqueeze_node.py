from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class UnsqueezeNode(PropagationNode):
    """Class that represents an onnx unsqueeze node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        unsqueeze_axes: list[int] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        if input_names is None:
            input_names = []
        op_type = "unsqueeze"
        super().__init__(node_id, node_name, op_type, input_names)

        self.unsqueeze_axes = unsqueeze_axes
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Unsqueeze an input tensor"""
        unsqueezed_tensor = tensor.unsqueeze(axes=self.unsqueeze_axes)

        if relevant_axes is None:
            relevant_axes = [False] * len(unsqueezed_tensor)
        if self.unsqueeze_axes is not None:
            for axis in self.unsqueeze_axes:
                relevant_axes.insert(True, axis)

        return unsqueezed_tensor, relevant_axes
