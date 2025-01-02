from typing import Any

from zigzag.utils import DiGraphWrapper

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.node import Node


class ONNXWorkload(DiGraphWrapper[Node]):
    """Represents a Workload Graph"""

    def __init__(self, **attr: Any):
        """Collect all the algorithmic workload information here."""
        super().__init__(**attr)  # type: ignore
        self.node_id_to_obj: dict[int, Node] = {}

    def add(self, node_id: int, node_obj: Node):
        """
        Add a node object to the ONNX workload graph.
        This can be a different object based on if it's an "accelerateable" node or not.
        """
        self.node_id_to_obj[node_id] = node_obj

        self.add_node(node_obj)
        edges: list[tuple[Node, Node]] = []
        for parent_id in node_obj.input_operand_source.values():
            parent_node_obj = self.node_id_to_obj[parent_id]
            edges.append((parent_node_obj, node_obj))
            self.add_edges_from(edges)


class ComputationNodeWorkload(DiGraphWrapper[ComputationNode]):
    """Workload graph with only ComputationNodes"""

    def get_sink_layer_ids(self):
        """Return the ids of layers where ALL sub-nodes have out-degree 0
        # TODO this might nog work yet! When there is intra-core tiling, edges between nodes in the same layer
        # TODO (with bits==0) are added, meaning some nodes have an out-degree > 0
        # TODO -> use get_real_nb_predecessors instead? or remove the empty intra-core edges?
        """
        out_degrees = self.out_degree()
        layer_ids = set(n.id for n, _ in out_degrees)
        sink_layer_ids = [
            curr_id
            for curr_id in layer_ids
            # x: (node, out_degree). Filter by id -> map to out_degree == 0 -> check if all are 0
            if all(map(lambda x: x[1] == 0, filter(lambda x: x[0].id == curr_id, out_degrees)))
        ]
        return sink_layer_ids

    def get_subgraph(self, nodes: list[ComputationNode]) -> "ComputationNodeWorkload":
        return self.subgraph(nodes)  # type: ignore
