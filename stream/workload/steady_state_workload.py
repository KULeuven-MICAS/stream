from typing import Any

import pydot  # 🔹 This was missing
from networkx.drawing.nx_pydot import to_pydot  # type: ignore
from zigzag.utils import DiGraphWrapper

from stream.workload.steady_state_computation import SteadyStateComputation
from stream.workload.steady_state_node import SteadyStateNode
from stream.workload.steady_state_rolling_buffer import SteadyStateRollingBuffer
from stream.workload.steady_state_tensor import SteadyStateTensor
from stream.workload.steady_state_transfer import SteadyStateTransfer


class SteadyStateWorkload(DiGraphWrapper[SteadyStateNode]):
    """Workload graph for steady state scheduling, supporting multiple node types."""

    def __init__(self, **attr: Any):
        super().__init__(**attr)

    def __str__(self) -> str:
        return (
            f"SteadyStateWorkload("
            f"computations={len(self.computation_nodes)}, "
            f"transfers={len(self.transfer_nodes)}, "
            f"tensors={len(self.tensor_nodes)}, "
            f"edges={self.number_of_edges()})"  # type: ignore
        )

    def __repr__(self) -> str:
        return str(self)

    def add(self, node_obj: SteadyStateNode):
        self.add_node(node_obj)
        # Edges should be added externally as needed

    def add_edge(self, u: SteadyStateNode, v: SteadyStateNode, **attrs: Any):
        """Add an edge between two nodes in the workload graph."""
        super().add_edges_from(
            [
                (u, v, attrs),
            ]
        )

    @property
    def computation_nodes(self) -> list[SteadyStateComputation]:
        """Return a list of all computation nodes in the workload."""
        return [node for node in self.nodes() if isinstance(node, SteadyStateComputation)]

    @property
    def transfer_nodes(self) -> list[SteadyStateTransfer]:
        """Return a list of all transfer nodes in the workload."""
        return [node for node in self.nodes() if isinstance(node, SteadyStateTransfer)]

    @property
    def tensor_nodes(self) -> list[SteadyStateTensor]:
        """Return a list of all tensor nodes in the workload."""
        return [node for node in self.nodes() if isinstance(node, SteadyStateTensor)]

    def get_subgraph(self, nodes: list[SteadyStateNode]) -> "SteadyStateWorkload":
        return self.subgraph(nodes)  # type: ignore

    def visualize_to_file(self, filepath: str = "workload_graph.png"):
        """Visualize the graph using Graphviz and save it to an image file.

        Nodes are laid out left to right by topological generation,
        and vertically grouped by chosen_resource_allocation (stacked top-to-bottom).
        """
        dot = to_pydot(self)

        # Set global graph layout left to right within clusters, vertical between clusters
        dot.set_rankdir("LR")
        dot.set_concentrate(True)

        # Group nodes by resource (vertical stacking of clusters)
        resource_to_nodes = {}
        for node in self.nodes():
            resource = getattr(node, "chosen_resource_allocation", -1)
            resource_to_nodes.setdefault(resource, []).append(node)

        # Sort clusters by resource id (top to bottom)
        sorted_resources = sorted(resource_to_nodes.keys(), key=lambda r: r.id if hasattr(r, "id") else -1)

        cluster_heads = []

        for resource in sorted_resources:
            nodes = resource_to_nodes[resource]

            subgraph = pydot.Cluster(
                graph_name=f"resource_{resource}",
                label=f"Resource {getattr(resource, 'id', resource)}",
                style="dashed",
                # No rank="same" so that global rankdir=LR applies inside the cluster
            )

            for node in nodes:
                n = dot.get_node(str(node))[0]
                subgraph.add_node(n)

            dot.add_subgraph(subgraph)

            if nodes:
                cluster_heads.append(str(nodes[0]))  # Use first node in cluster for invisible edge anchor

        # Add invisible edges between cluster heads to stack clusters vertically
        for i in range(len(cluster_heads) - 1):
            upper = cluster_heads[i]
            lower = cluster_heads[i + 1]
            dot.add_edge(pydot.Edge(upper, lower, style="invis"))

        # Customize node appearance
        for node in self.nodes():
            n = dot.get_node(str(node))[0]
            n.set_label(f"{node.node_name}")
            n.set_shape("box")
            n.set_style("filled")
            n.set_fillcolor(
                "#a2d5f2"
                if isinstance(node, SteadyStateComputation)
                else (
                    "#ffcb9a"
                    if isinstance(node, SteadyStateTransfer)
                    else (
                        "#eaff76e1"
                        if isinstance(node, SteadyStateRollingBuffer)
                        else "#c2f0c2" if isinstance(node, SteadyStateTensor) else "#eeeeee"
                    )
                )
            )

        # Save to file
        dot.write_png(filepath)
        print(f"Graph saved to {filepath}")
