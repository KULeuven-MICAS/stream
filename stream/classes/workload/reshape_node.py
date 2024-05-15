from typing import Any
import numpy as np
from stream.classes.workload.node import Node
from zigzag.datatypes import LayerOperand
from zigzag.workload.LayerNodeABC import LayerNodeABC


class ReshapeNode(Node, LayerNodeABC):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        shape: list[int],
        input_names: list[str],
        output_names: list[str],
    ) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors: The id of this node's parent.
            shape: The output tensor's shape.
            input_names The input names of this node.
            output_names: The output names of this node.
        """
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type="reshape",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            core_allocation=[-1],
            input_names=input_names,
            output_names=output_names,
        )

        self.shape = shape
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def reshape_operand_tensor(self, tensor: np.ndarray[Any, Any]):
        """Reshape the tensor back to the representation needed for producer/consumer."""
        new_shape = self.shape
        if not new_shape:
            new_shape = tensor.shape
        return np.reshape(tensor, new_shape)
