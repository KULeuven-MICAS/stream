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
        input_names: list[str] = [],
    ) -> None:
        PropagationNode.__init__(self, node_id, node_name, op_type, input_names)
        DummyNodeZigZag.__init__(
            self,
            node_id=node_id,
            predecessors=predecessors,
            node_type=op_type,
            node_name=node_name,
        )

    def propagate(self, tensor: NodeTensor, next_node: Node | None = None) -> NodeTensor:
        return tensor
