from typing import Any

import numpy as np
from zigzag.datatypes import LayerOperand
from zigzag.workload.LayerNodeABC import LayerNodeABC

from stream.classes.workload.node import Node


class GatherNode(Node, LayerNodeABC):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        gather_axis: int,
        gather_indices: int | list[int],
        input_names: list[str],
        output_names: list[str],
    ) -> None:
        """Initialize the GatherNode

        Args:
            predecessors: The id of this node's parent.
            gather_axis: Which axis to gather on.
            gather_indices: Indices of elements to be gathered.
            input_names The input names of this node.
            output_names: The output names of this node.
        """
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type="gather",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
            input_names=input_names,
            output_names=output_names,
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

        self.gather_axis = gather_axis
        self.gather_indices = gather_indices
        match len(predecessors):
            case 0:
                self.input_operand_source = {}
            case 1:
                self.input_operand_source = {LayerOperand("I"): predecessors[0]}
            case 2:
                # `indices` (the second input) are considered as inputs
                self.input_operand_source = {LayerOperand("W"): predecessors[0], LayerOperand("I"): predecessors[1]}
            case _:
                raise ValueError("More than two inputs for GatherNode")

    def gather_operand_tensor(self, tensor: np.ndarray[Any, Any]):
        """Perform gather operation on the tensor."""
        return np.take(tensor, self.gather_indices, axis=self.gather_axis)
