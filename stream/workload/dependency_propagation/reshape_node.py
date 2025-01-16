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
        input_names: list[str] = [],
    ) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors: The id of this node's parent.
            shape: The output tensor's shape.
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
        """
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
        relevant_axes: list[bool] = [],
    ) -> tuple[NodeTensor, list[bool]]:
        """Reshape the tensor back to the representation needed for producer/consumer."""
        new_shape = self.shape
        if not new_shape:
            return tensor

        if not self.allow_zero:
            new_shape = tuple(x for x in new_shape if x != 0)

        shape_change_axes = [
            i for i in range(len(new_shape)) if i >= len(tensor.tensor_shape) or new_shape[i] != tensor.tensor_shape[i]
        ]
        for axis in shape_change_axes:
            if axis >= len(relevant_axes):
                relevant_axes.append(True)
            else:
                relevant_axes[axis] = True

        return tensor.reshape(new_shape), relevant_axes
