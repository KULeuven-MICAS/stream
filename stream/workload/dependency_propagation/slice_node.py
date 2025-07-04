from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class SliceNode(PropagationNode):
    """Class that represents an onnx Slice node."""

    def __init__(  # noqa: PLR0913
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        starts: list[int],
        ends: list[int],
        axes: list[int],
        steps: list[int],
        output_names: list[str],
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the SliceNode
        Slice the tensor at axis `axis`. The sizes are given by `Slices`. `len(Slices)` is the number of output nodes.

        Args:
            predecessors: The id of this node's parent.
            axis: axis in which to Slice
            Slices: sizes of the output Slices in the given axis
            output_names: the node names that correspond to the Slices
        """
        if input_names is None:
            input_names = []
        op_type = "Slice"
        super().__init__(node_id, node_name, op_type, input_names)

        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}
        self.output_names = output_names

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Slice the tensor.
        Currently assumes only one slice is created."""
        sliced_tensor = tensor.slice(starts=self.starts[0], ends=self.ends[0], axis=self.axes[0], steps=self.steps[0])
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        relevant_axes[self.axes[0]] = True
        return sliced_tensor, relevant_axes
