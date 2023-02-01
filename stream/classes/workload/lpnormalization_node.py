import numpy as np
from stream.classes.workload.node import Node


class LpNormalizationNode(Node):
    """Class that represents an onnx LpNormalization node.
    """
    def __init__(self, predecessors, input_names, output_names) -> None:
        """Initialize the LpNormalization node.

        Args:
            predecessors (list): The predecessors of this node.
            input_names (list) The input names of this node.
            output_names (list): The output names of this node.
        """
        super().__init__("lpnormalization", energy=0, runtime=0, core_allocation=-1, input_names=input_names, output_names=output_names)
        self.input_operand_source = {'I': predecessors}

    #def lpnormalization(self, input_tensor):
    #    """Reshape an input tensor

    #    Args:
    #        input_tensor (np.ndarray): The input tensor
    #    """
    #    return softmax(input_tensor,axis=-1)

    def lpnormalization_operand_tensor(self, tensor):
        """Propagate the input tensor dependencies.
        """
        temp = tensor.copy()
        size_hor = np.size(temp,0)
        size_ver = np.size(temp,1)
        for i in range(size_hor):
            the_list = temp[i][0]
            for j in range(1, size_ver):
                the_list = the_list.union(temp[i][j])
            for j in range(size_ver):
                temp[i][j] = the_list
        return temp