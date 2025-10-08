from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class PadNode(PropagationNode):
    """Class that represents an onnx Pad node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        padding: tuple[int, ...],
        allow_zero: bool = False,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the PadNode

        Args:
            predecessor: The id of this node's parent.
            padding: Padding to add to the input tensor
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
        """
        if input_names is None:
            input_names = []
        op_type = "pad"
        super().__init__(node_id, node_name, op_type, input_names)

        self.allow_zero = allow_zero
        self.padding = padding
        if len(predecessor) == 0:
            self.input_operand_source = {}
        else:
            self.input_operand_source = {Constants.LAYER_OP_I: predecessor[0]}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Perform gather operation on the tensor."""

        for axis in range(0, len(self.padding) // 2):
            if self.padding[2 * axis] != 0 or self.padding[2 * axis + 1] != 0:
                relevant_axes[axis] = True

        return tensor.pad(self.padding), relevant_axes
