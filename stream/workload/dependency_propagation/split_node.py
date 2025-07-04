from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
from stream.workload.computation.computation_node import GeneratedComputationNode
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class SplitNode(PropagationNode):
    """Class that represents an onnx Split node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        axis: int,
        splits: list[int],
        output_names: list[str],
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the SplitNode
        Split the tensor at axis `axis`. The sizes are given by `splits`. `len(splits)` is the number of output nodes.

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which to split
            splits: sizes of the output splits in the given axis
            output_names: the node names that correspond to the splits, used to determine propagation flow
        """
        if input_names is None:
            input_names = []
        op_type = "split"
        super().__init__(node_id, node_name, op_type, input_names)

        self.axis = axis
        self.splits = splits
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}
        self.output_names = output_names

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Split the tensor back to the representation needed for producer/consumer."""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        assert next_node is not None

        index = self.find_split_index(next_node)

        if index >= len(self.splits):
            raise ValueError(f"Found slice index {index} for next node {next_node} exceeds slice dimensions")

        start_idx = sum(self.splits[:index])
        end_idx = start_idx + self.splits[index]
        output_tensor = tensor.slice(start_idx, end_idx, axis=self.axis)

        # Update the relevant_dims with the axis involved in the split
        relevant_axes[self.axis] = True

        assert len(tensor.tensor_shape) == len(output_tensor.tensor_shape)
        return output_tensor, relevant_axes

    def find_split_index(self, next_node: Node):
        """Given the next node that comes after this split node, return the index of this node's splitted outputs that
        corresponds to the next node's input"""
        if isinstance(next_node, GeneratedComputationNode):
            # Assume that the slice index corresponds to the order in which the next node was generated
            return next_node.gen_id

        # Find which split part corresponds to the input of the next node
        try:
            index = next(i for i, output_name in enumerate(self.output_names) if output_name in next_node.input_names)
            return index
        except StopIteration as exc:
            raise ValueError(
                f"Cannot find outputs {self.output_names} of {self.name} in next inputs {next_node.input_names}"
            ) from exc
