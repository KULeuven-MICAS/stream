import numpy as np
from stream.classes.workload.node import Node


class FlattenNode(Node):
    """Class that represents an onnx Flatten node."""

    def __init__(self, id, predecessors, axis, input_names, output_names) -> None:
        """Initialize the FlattenNode

        Args:
            shape (list): The output tensor's shape.
        """
        super().__init__(
            "flatten",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            core_allocation=-1,
            input_names=input_names,
            output_names=output_names,
        )
        self.id = id
        self.axis = axis
        self.input_operand_source = {"I": predecessors}

    def flatten(self, input_tensor):
        """Reshape an input tensor

        Args:
            input_tensor (np.ndarray): The input tensor
        """
        shape = input_tensor.shape
        new_shape = (
            (1, -1)
            if self.axis == 0
            else (np.prod(shape[0 : self.axis]).astype(int), -1)
        )  # taken from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-51
        return np.reshape(input_tensor, new_shape)
