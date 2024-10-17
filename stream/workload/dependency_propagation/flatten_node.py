import numpy as np
from zigzag.datatypes import LayerOperand
from zigzag.workload.layer_node_abc import LayerNodeABC

from stream.node_tensor import NodeTensor
from stream.workload.node import Node


class FlattenNode(Node, LayerNodeABC):
    """Class that represents an onnx Flatten node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int | None,
        axis: int | None,
    ) -> None:
        """Initialize the FlattenNode

        Args:
            shape (list): The output tensor's shape.
        """
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type="flatten",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        self.axis = axis
        if predecessor is not None:
            self.input_operand_source = {LayerOperand("I"): predecessor}

    def flatten(self, input_tensor: NodeTensor) -> NodeTensor:
        """Reshape an input tensor

        Args:
            input_tensor (np.ndarray): The input tensor
        """
        shape = input_tensor.tensor_shape
        # taken from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-51
        new_shape = (1, -1) if self.axis == 0 else (np.prod(shape[0 : self.axis]).astype(int), -1)
        return input_tensor.reshape(new_shape)
