"""Weisfeiler-Leman colour refinement over the workload DAG."""

from __future__ import annotations

import hashlib

from stream.workload.node import ComputationNode, Node
from stream.workload.node_key import node_key
from stream.workload.workload import Workload


def _initial_colour(node: Node) -> str:
    if isinstance(node, ComputationNode):
        return f"C:{node_key(node)}"
    return f"N:{type(node).__name__}"


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def refine_colours(workload: Workload, rounds: int = 3) -> dict[Node, str]:
    """Return the WL colour of every node after ``rounds`` refinement rounds."""
    colours: dict[Node, str] = {node: _initial_colour(node) for node in workload.nodes}
    for _ in range(rounds):
        refined: dict[Node, str] = {}
        for node in workload.nodes:
            predecessors = sorted(colours[p] for p in workload.predecessors(node))
            successors = sorted(colours[s] for s in workload.successors(node))
            refined[node] = _hash(f"{colours[node]}|<{','.join(predecessors)}|>{','.join(successors)}")
        colours = refined
    return colours
