from zigzag.workload.layer_node import LayerNodeAttributes

from stream.classes.workload.computation_node import ComputationNode


class PoolingNode(ComputationNode):
    def __init__(
        self,
        node_id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
        input_names: list[str],
        output_names: list[str],
    ):
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            node_attr=node_attr,
            input_names=input_names,
            output_names=output_names,
            op_type="pooling",
        )
        self.type = "pooling"
