import numpy as np
from stream.classes.workload.node import Node


class ReshapeNode(Node):
    """Class that represents an onnx Reshape node.
    """
    def __init__(self, predecessors, shape, input_names, output_names) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors (list): The predecessors of this node.
            shape (list): The output tensor's shape.
            input_names (list) The input names of this node.
            output_names (list): The output names of this node.
        """
        super().__init__("reshape", energy=0, runtime=0, core_allocation=-1, input_names=input_names, output_names=output_names)
        self.shape = shape
        self.input_operand_source = {'I': predecessors}

    def reshape_operand_tensor(self, tensor):
        """Reshape the tensor back to the representation needed for producer/consumer.
        """
        new_shape = self.shape
        if not new_shape:
            new_shape = tensor.shape
        return np.reshape(tensor, new_shape)