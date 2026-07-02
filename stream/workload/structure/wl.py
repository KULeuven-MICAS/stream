"""Weisfeiler-Leman colour refinement over the workload DAG.

Each node starts from its :func:`~stream.workload.node_key.node_key` (compute nodes) or a role tag
(edges/transfers); each round replaces a node's colour with a hash of its own colour and the sorted
multiset of its predecessors' and successors' colours (edge direction kept). After ``rounds`` rounds,
nodes with equal colours have identical local structure up to that radius -- so a change deep inside
one repeated block propagates outward and splits the affected neighbourhood's colours.
"""

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
