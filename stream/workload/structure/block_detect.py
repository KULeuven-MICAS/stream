"""Repeated-block detection via WL colour classes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from stream.workload.node import ComputationNode
from stream.workload.structure.wl import refine_colours
from stream.workload.workload import Workload


@dataclass(frozen=True)
class BlockClass:
    colour: str
    nodes: tuple[ComputationNode, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.nodes)


def find_repeated_blocks(workload: Workload, rounds: int = 0) -> list[BlockClass]:
    """Computation-node equivalence classes with multiplicity > 1, most-repeated first
    (``rounds>0`` adds WL-neighbourhood distinction)."""
    colours = refine_colours(workload, rounds)
    groups: dict[str, list[ComputationNode]] = defaultdict(list)
    for node in workload.get_computation_nodes():
        groups[colours[node]].append(node)
    classes = [BlockClass(colour=colour, nodes=tuple(nodes)) for colour, nodes in groups.items()]
    return sorted((c for c in classes if c.multiplicity > 1), key=lambda c: (-c.multiplicity, c.colour))
