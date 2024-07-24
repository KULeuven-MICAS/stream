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
        allow_zero: bool = False,
    ) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors: The id of this node's parent.
            shape: The output tensor's shape.
            input_names The input names of this node.
            output_names: The output names of this node.
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
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

        self.allow_zero = allow_zero
        self.shape = shape
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def reshape_operand_tensor(self, tensor: np.ndarray[Any, Any]):
        """Reshape the tensor back to the representation needed for producer/consumer."""
        new_shape = self.shape
        if not new_shape:
            new_shape = tensor.shape

        if not self.allow_zero:
            new_shape = [x for x in new_shape if x != 0]
        return np.reshape(tensor, new_shape)
