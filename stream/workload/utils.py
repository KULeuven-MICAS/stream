from typing import TYPE_CHECKING

from networkx import DiGraph

if TYPE_CHECKING:
    from stream.workload.computation.computation_node import ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload


def prune_workload(G: DiGraph, keep_types=[]):
    """Return a pruned workload graph with only nodes of type in 'keep_types'."""
    while any(any(not isinstance(node, keep_type) for keep_type in keep_types) for node in G.nodes()):
        G0 = G.copy()
        for node in G.nodes():
            if any(not isinstance(node, keep_type) for keep_type in keep_types):
                assert G.is_directed()
                in_edges_containing_node = list(G0.in_edges(node))
                out_edges_containing_node = list(G0.out_edges(node))
                for in_src, _ in in_edges_containing_node:
                    for _, out_dst in out_edges_containing_node:
                        G0.add_edge(in_src, out_dst)
                G0.remove_node(node)
                break
        G = G0
    return G


def get_real_successors(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(n for n in g.successors(node) if n.id != node.id)
