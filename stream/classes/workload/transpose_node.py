import numpy as np
from stream.classes.workload.node import Node


class TransposeNode(Node):
    """Class that represents an onnx Transpose node.
    """
    def __init__(self, predecessors, input_names, output_names) -> None:
        """Initialize the TransposeNode

        Args:
            predecessors (list): The predecessors of this node.
            input_names (list) The input names of this node.
            output_names (list): The output names of this node.
        """
        super().__init__("transpose", energy=0, runtime=0, core_allocation=-1, input_names=input_names, output_names=output_names)
        self.input_operand_source = {'I': predecessors}

    def transpose(self, input_tensor):
        """Transpose an input tensor.

        Args:
            input_tensor (np.ndarray): The input tensor
        """
        return np.transpose(input_tensor)
