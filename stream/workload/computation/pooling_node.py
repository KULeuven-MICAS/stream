from zigzag.workload.layer_node import LayerNodeAttributes

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import InterCoreMappingAttributes


class PoolingNode(ComputationNode):
    """TODO this node can be replaced by instantiating ComputationNode directly"""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
        mapping_attr: InterCoreMappingAttributes,
        input_names: list[str] | None = None,
    ):
        if input_names is None:
            input_names = []
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            node_attr=node_attr,
            mapping_attr=mapping_attr,
            op_type="pooling",
            input_names=input_names,
        )
