from zigzag.datatypes import LayerOperand
from zigzag.workload.layer_node_abc import LayerNodeABC

from stream.utils import NodeTensor
from stream.workload.node import Node


class TransposeNode(Node, LayerNodeABC):
    """Class that represents an onnx Transpose node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        permute_axes: list[int] | None = None,
    ) -> None:
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type="reshape",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

        self.permute_axes = permute_axes
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def transpose(self, input_tensor: NodeTensor) -> NodeTensor:
        """Transpose an input tensor.

        Args:
            input_tensor (np.ndarray): The input tensor
        """
        return input_tensor.transpose(axes=self.permute_axes)
