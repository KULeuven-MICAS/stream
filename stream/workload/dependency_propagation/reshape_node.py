from yaml import Node
from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode


class ReshapeNode(PropagationNode):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        shape: tuple[int, ...],
        allow_zero: bool = False,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors: The id of this node's parent.
            shape: The output tensor's shape.
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
        """
        if input_names is None:
            input_names = []
        op_type = "reshape"
        super().__init__(node_id, node_name, op_type, input_names)

        self.allow_zero = allow_zero
        self.shape = shape
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Reshape the tensor back to the representation needed for producer/consumer."""
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        new_shape = self.shape
        if not new_shape:
            return tensor, relevant_axes

        if not self.allow_zero:
            new_shape = tuple(x for x in new_shape if x != 0)

        relevant_axes = self.update_relevant_axes(relevant_axes, tensor.tensor_shape, new_shape)

        return tensor.reshape(new_shape), relevant_axes

    def update_relevant_axes(self, relevant_axes: list[bool], old_shape: tuple[int, ...], new_shape: tuple[int, ...]):
        new_relevant_axes = len(new_shape) * [False]

        for idx, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape, strict=False)):
            new_relevant_axes[idx] = relevant_axes[idx] if idx < len(relevant_axes) else False
            if old_dim != new_dim:
                new_relevant_axes[idx] = True

        return new_relevant_axes
