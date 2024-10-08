from zigzag.workload.dummy_node import DummyNode as DummyNodeZigZag

from stream.workload.node import Node


class DummyNode(DummyNodeZigZag, Node):
    """DummyNode of an onnx operator that is not import for finer graph generation or for cost estimation,
    but plays a role because of the passing of the input and output tensors.
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessors: list[int],
        op_type: str = "dummy",
    ) -> None:
        DummyNodeZigZag.__init__(
            self,
            node_id=node_id,
            predecessors=predecessors,
            node_type=op_type,
            node_name=node_name,
        )
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type=op_type,
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
