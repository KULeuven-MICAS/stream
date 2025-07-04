from abc import abstractmethod

from zigzag.workload.layer_node_abc import LayerNodeABC

from stream.node_tensor import NodeTensor
from stream.workload.node import Node


class PropagationNode(Node, LayerNodeABC):
    """Stream node that does not perform computations and is not mapped on hardware, but propagates dependencies
    between nodes"""

    def __init__(self, node_id: int, node_name: str, op_type: str, input_names: list[str]):
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type=op_type,
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
            input_names=input_names,
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

    @abstractmethod
    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]: ...
