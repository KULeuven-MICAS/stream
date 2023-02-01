from stream.classes.workload.node import Node


class ElementwiseNode(Node):
    """Class that represents an onnx Reshape node.
    """
    def __init__(self, type, name, predecessors, input_names, output_names) -> None:
        """Initialize the ReshapeNode.
        """
        super().__init__(type, energy=0, runtime=0, core_allocation=-1, input_names=input_names, output_names=output_names)
        self.name = name
        self.input_operand_source = {'I': predecessors}

    def join(self, tensor1, tensor2):
        """Join each position in the two tensors to propagate the dependencies (each position should contain a set).

        Args:
            tensor1 (np.ndarray): The first input tensor
            tensor2 (np.ndarray): The second input tensor
        """
        return tensor1 | tensor2
