from typing import Any
import numpy as np
from stream.classes.workload.node import Node
from zigzag.datatypes import LayerOperand
from zigzag.workload.LayerNodeABC import LayerNodeABC


class FlattenNode(Node, LayerNodeABC):
    """Class that represents an onnx Flatten node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int | None,
        axis: int | None,
        input_names: list[str],
        output_names: list[str],
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
            input_names=input_names,
            output_names=output_names,
        )
        self.axis = axis
        if predecessor is not None:
            self.input_operand_source = {LayerOperand("I"): predecessor}

    def flatten(self, input_tensor: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Reshape an input tensor

        Args:
            input_tensor (np.ndarray): The input tensor
        """
        shape = input_tensor.shape
        # taken from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-51
        new_shape = (1, -1) if self.axis == 0 else (np.prod(shape[0 : self.axis]).astype(int), -1)
        return np.reshape(input_tensor, new_shape)
