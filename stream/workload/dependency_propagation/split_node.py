import numpy as np
from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
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
        input_names: list[str] = [],
    ) -> None:
        """Initialize the SplitNode
        Split the tensor at axis `axis`. The sizes are given by `splits`. `len(splits)` is the number of output nodes.

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which to split
            splits: sizes of the output splits in the given axis
            output_names: the node names that correspond to the splits
        """
        assert len(splits) == len(output_names)
        op_type = "split"
        super().__init__(node_id, node_name, op_type, input_names)

        self.axis = axis
        self.splits = splits
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}
        self.output_names = output_names

    def propagate(self, tensor: NodeTensor, next_node: Node, relevant_axes: list[bool]):
        """Split the tensor back to the representation needed for producer/consumer."""

        # Numpy requires the indices where to split instead of the sizes of the resulting splits
        split_indices = list(np.cumsum(self.splits)[:-1])
        output_tensors = tensor.split(split_indices, axis=self.axis)

        # Find which split part corresponds to the input of the next node
        try:
            index = next(i for i, output_name in enumerate(self.output_names) if output_name in next_node.input_names)
        except StopIteration:
            raise ValueError(
                f"Cannot find this nodes' ({self.name}) outputs {self.output_names} in next nodes' inputs {next_node.input_names}"
            )

        # Update the relevant_dims with the axis involved in the split
        relevant_axes[self.axis] = True

        output_tensor = output_tensors[index]
        return output_tensor, relevant_axes
