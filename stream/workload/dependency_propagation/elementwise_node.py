from zigzag.datatypes import LayerOperand

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class ElementwiseNode(PropagationNode):
    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        input_names: list[str],
    ) -> None:
        op_type = "elementwise"
        super().__init__(node_id, node_name, op_type, input_names)
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def join(self, tensor1, tensor2):
        """Join each position in the two tensors to propagate the dependencies (each position should contain a set).

        Args:
            tensor1 (np.ndarray): The first input tensor
            tensor2 (np.ndarray): The second input tensor
        """
        return tensor1 | tensor2

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)
        return tensor, relevant_axes
