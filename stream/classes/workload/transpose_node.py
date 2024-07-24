from typing import Any
import numpy as np
from stream.classes.workload.node import Node
from zigzag.datatypes import LayerOperand
from zigzag.workload.LayerNodeABC import LayerNodeABC


class TransposeNode(Node, LayerNodeABC):
    """Class that represents an onnx Transpose node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        input_names: list[str],
        output_names: list[str],
        permute_axes: list[int] | None = None,
    ) -> None:
        """Initialize the TransposeNode

        Args:
            predecessors: The predecessors of this node.
            input_names The input names of this node.
            output_names: The output names of this node.
        """

        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type="reshape",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
            input_names=input_names,
            output_names=output_names,
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

        self.permute_axes = permute_axes
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def transpose(self, input_tensor: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Transpose an input tensor.

        Args:
            input_tensor (np.ndarray): The input tensor
        """
        return np.transpose(input_tensor, axes=self.permute_axes)
