from typing import TYPE_CHECKING

from networkx import DiGraph

if TYPE_CHECKING:
    from stream.workload.computation.computation_node import ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload


def prune_workload(g: DiGraph, keep_types=None):
    """Return a pruned workload graph with only nodes of type in 'keep_types'."""
    if keep_types is None:
        keep_types = []
    while any(any(not isinstance(node, keep_type) for keep_type in keep_types) for node in g.nodes()):
        g_copy = g.copy()
        for node in g.nodes():
            if any(not isinstance(node, keep_type) for keep_type in keep_types):
                assert g.is_directed()
                in_edges_containing_node = list(g_copy.in_edges(node))  # type: ignore
                out_edges_containing_node = list(g_copy.out_edges(node))  # type: ignore
                for in_src, _ in in_edges_containing_node:
                    for _, out_dst in out_edges_containing_node:
                        g_copy.add_edge(in_src, out_dst)
                g_copy.remove_node(node)
                break
        g = g_copy  # type: ignore
    return g


def get_real_predecessors(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(n for n in g.predecessors(node) if n.id != node.id)


def get_real_successors(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(n for n in g.successors(node) if n.id != node.id)


def get_real_in_edges(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(e for e in g.in_edges(node, data=True) if e[0].id != node.id)


def get_real_out_edges(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(e for e in g.out_edges(node, data=True) if e[1].id != node.id)
