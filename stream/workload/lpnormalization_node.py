import numpy as np
from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.node import Node


class LpNormalizationNode(Node):
    """Class that represents an onnx LpNormalization node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
    ) -> None:
        """Initialize the LpNormalization node.

        Args:
            predecessors (list): The predecessors of this node.
        """
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type="lpnormalization",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def lpnormalization_operand_tensor(self, tensor: NodeTensor) -> NodeTensor:
        """Propagate the input tensor dependencies."""
        raise NotImplementedError("TODO: make sure this is bug-free after transformer changes")
        temp = tensor.copy()
        size_hor = np.size(temp, 0)
        size_ver = np.size(temp, 1)
        for i in range(size_hor):
            the_list = temp[i][0]
            for j in range(1, size_ver):
                the_list = the_list.union(temp[i][j])
            for j in range(size_ver):
                temp[i][j] = the_list
        return temp
