from zigzag.workload.dummy_node import DummyNode as DummyNodeZigZag

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class DummyNode(DummyNodeZigZag, PropagationNode):
    """DummyNode of an onnx operator that is not import for finer graph generation or for cost estimation,
    but plays a role because of the passing of the input and output tensors.
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        op_type: str = "dummy",
        input_names: list[str] | None = None,
    ) -> None:
        if input_names is None:
            input_names = []
        PropagationNode.__init__(self, node_id, node_name, op_type, input_names)
        DummyNodeZigZag.__init__(
            self,
            node_id=node_id,
            predecessors=predecessors,
            node_type=op_type,
            node_name=node_name,
        )

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
