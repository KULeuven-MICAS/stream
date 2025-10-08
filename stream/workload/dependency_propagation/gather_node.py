import numpy as np
from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class GatherNode(PropagationNode):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        gather_axis: int,
        gather_indices: int | list[int],
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the GatherNode

        Args:
            predecessors: The id of this node's parent.
            gather_axis: Which axis to gather on.
            gather_indices: Indices of elements to be gathered.
        """
        if input_names is None:
            input_names = []
        op_type = "gather"
        super().__init__(node_id, node_name, op_type, input_names)

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

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Perform gather operation on the tensor."""
        new_rank = len(tensor.tensor_shape) + (np.array(self.gather_indices).ndim - 1)
        if relevant_axes is None:
            relevant_axes = [False] * new_rank

        if len(relevant_axes) < new_rank:
            relevant_axes.insert(self.gather_axis, [True] * (new_rank - len(relevant_axes)))
            for i in range(0, np.array(self.gather_indices).ndim):
                relevant_axes[self.gather_axis + i] = True
        if len(relevant_axes) > new_rank:
            # Can happen when gather_indices is a scalar
            del relevant_axes[self.gather_axis]

        return tensor.gather(self.gather_indices, axis=self.gather_axis), relevant_axes
