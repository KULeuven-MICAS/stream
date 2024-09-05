from zigzag.workload.layer_node import LayerNodeAttributes

from stream.workload.computation_node import ComputationNode


class PoolingNode(ComputationNode):
    def __init__(
        self,
        node_id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
    ):
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            node_attr=node_attr,
            op_type="pooling",
        )
