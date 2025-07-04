import numpy as np
from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class FlattenNode(PropagationNode):
    """Class that represents an onnx Flatten node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int | None,
        axis: int | None,
        input_names: list[str],
    ) -> None:
        """Initialize the FlattenNode

        Args:
            shape: The output tensor's shape.
        """
        op_type = "flatten"
        super().__init__(node_id, node_name, op_type, input_names)

        self.axis = axis
        if predecessor is not None:
            self.input_operand_source = {LayerOperand("I"): predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Reshape an input tensor"""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        shape = tensor.tensor_shape
        # taken from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-51
        new_shape = (1, -1) if self.axis == 0 else (np.prod(shape[0 : self.axis]).astype(int), -1)
        # All axes will be relevant in case of a flatten operation
        relevant_axes = [True] * len(new_shape)
        return tensor.reshape(new_shape), relevant_axes
