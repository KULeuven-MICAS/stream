import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import combinations, product
from math import prod
from typing import TYPE_CHECKING

import networkx as nx

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

Box = tuple[tuple[int, int], ...]
"""Inclusive index range per axis of the data a demand moves; equal boxes are the same data."""

_HOP = 1e-9
"""Cost of a hop beside the cycles a link adds, so equally fast routes prefer the shorter."""


@dataclass(frozen=True)
class DemandItem:
    """Data the target at ``target`` reads, ``box`` of the tensor, held by each source at ``sources``, any one of which
    can serve it; or, if ``reduce``, of which each holds a partial sum that the target needs all of."""

    target: int
    sources: tuple[int, ...]
    box: Box
    reduce: bool = False


@dataclass(frozen=True, slots=True)
class MulticastPathPlan:
    """A transfer's route: which source serves which target (``pairs``, by position), the links the data crosses and
    the share of the transfer's bits each carries, one copy of the data per source however many targets it reaches."""

    sources: tuple["Core", ...]
    targets: tuple["Core", ...]
    total_hops_objective: int
    links_used: tuple["CommunicationLink", ...]
    pairs: tuple[tuple[int, int], ...] = ()
    link_shares: tuple[tuple["CommunicationLink", float], ...] = field(default=(), compare=False)


def box_bits(boxes: Iterable[Box], bitwidth: float) -> float:
    """Bits the ``boxes`` cover together: distinct data counted once."""
    unique = list(dict.fromkeys(boxes))
    return _covered(unique) * bitwidth


def _covered(boxes: list[Box]) -> int:
    """Points the inclusive boxes cover together, sweeping the first axis between the boxes' edges."""
    if not boxes:
        return 0
    if len(boxes) == 1 or all(_disjoint(a, b) for a, b in combinations(boxes, 2)):
        return sum(prod(hi - lo + 1 for lo, hi in box) for box in boxes)
    if len(boxes[0]) == 1:
        spans = sorted((box[0][0], box[0][1]) for box in boxes)
        total, end = 0, -1
        for lo, hi in spans:
            if hi > end:
                total += hi - max(lo, end + 1) + 1
                end = hi
        return total
    edges = sorted({box[0][0] for box in boxes} | {box[0][1] + 1 for box in boxes})
    total = 0
    for lo, nxt in zip(edges, edges[1:], strict=False):
        inside = [box[1:] for box in boxes if box[0][0] <= lo and box[0][1] >= nxt - 1]
        total += (nxt - lo) * _covered(list(dict.fromkeys(inside)))
    return total


def _disjoint(a: Box, b: Box) -> bool:
    return any(ha < lb or hb < la for (la, ha), (lb, hb) in zip(a, b, strict=True))


class CommunicationManager:
    """
    Manages communication events and link usage between cores, including bandwidth normalization and event creation.
    Handles both data transfers and link blocking for memory constraints.
    """

    shortest_paths: dict[tuple[Core, Core], list[Core]]

    def __init__(self, accelerator: "Accelerator") -> None:
        self.accelerator = accelerator
        self.all_pair_links = self.get_all_links_for_all_core_pairs()
        self._transfer_plans: dict[tuple, tuple[MulticastPathPlan, ...]] = {}

    def get_all_links_for_all_core_pairs(self) -> dict[tuple[Core, Core], tuple[tuple[CommunicationLink, ...], ...]]:
        """The links of every shortest path between each pair of cores one can reach from the other."""
        graph = self.accelerator.cores
        links: dict[tuple[Core, Core], tuple[tuple[CommunicationLink, ...], ...]] = {}
        for sender, receiver in product(self.accelerator.core_list, self.accelerator.core_list):
            if nx.has_path(graph, sender, receiver):
                paths = nx.all_shortest_paths(graph, sender, receiver)
                links[(sender, receiver)] = tuple(
                    tuple(graph.edges[edge]["cl"] for edge in zip(path, path[1:], strict=False)) for path in paths
                )
        return links

    def _adjacency(self) -> dict[Core, list[tuple[Core, CommunicationLink]]]:
        """Each core's outgoing links, computed once per accelerator."""
        adjacency = self.accelerator.__dict__.get("_route_adjacency")
        if adjacency is None:
            adjacency = {core: [] for core in self.accelerator.core_list}
            for u, v, data in self.accelerator.cores.edges(data=True):
                adjacency[u].append((v, data["cl"]))
            for links in adjacency.values():
                links.sort(key=lambda edge: edge[0].id)
            self.accelerator.__dict__["_route_adjacency"] = adjacency
        return adjacency

    def _reverse_adjacency(self) -> dict[Core, list[tuple[Core, CommunicationLink]]]:
        """Each core's incoming links, computed once per accelerator."""
        reverse = self.accelerator.__dict__.get("_route_reverse_adjacency")
        if reverse is None:
            reverse = {core: [] for core in self.accelerator.core_list}
            for u, edges in self._adjacency().items():
                for v, link in edges:
                    reverse[v].append((u, link))
            self.accelerator.__dict__["_route_reverse_adjacency"] = reverse
        return reverse

    def get_possible_transfer_plan(
        self,
        src_allocs: Iterable["Core"],
        dst_allocs: Iterable["Core"],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[["Core", "Core"], bool] = lambda one, other: False,
    ) -> tuple[MulticastPathPlan, ...]:
        """The route of a transfer between these cores that moves ``demand``, planned once per demand."""
        key = (tuple(src_allocs), tuple(dst_allocs), demand, bitwidth)
        plans = self._transfer_plans.get(key)
        if plans is None:
            plans = self._transfer_plans[key] = (self.route(key[0], key[1], demand, bitwidth, shares_memory),)
        return plans

    def route(
        self,
        sources: tuple["Core", ...],
        targets: tuple["Core", ...],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[["Core", "Core"], bool],
    ) -> MulticastPathPlan:
        """Serve each demand from the nearest source holding it, then grow one Steiner tree per source over the
        targets it serves (the shortest-path heuristic of Takahashi and Matsuyama), the largest source first and each
        link weighted by the cycles it would then be busy, so later trees avoid what earlier ones load. A link carries
        the union of the data its tree sends past it. A target in its source's memory needs no route; one that reaches
        its source's memory otherwise, as a neighbouring tile does, is reached on a route that moves none of it. A core
        sharing a memory sends and receives through the core owning it. Partial sums are then reduced on the way, see
        :meth:`_reduce`. Each link's share is of all the data the transfer delivers, whether it moves or is already in
        its target's memory."""
        served: dict[int, list[DemandItem]] = {}
        in_place: dict[int, set[Core]] = {}
        pairs: set[tuple[int, int]] = set()
        reduced: dict[int, list[DemandItem]] = {}
        delivered: list[Box] = []
        for item in demand:
            target = targets[item.target]
            if item.reduce:
                pairs.update((s, item.target) for s in item.sources)
                reduced.setdefault(item.target, []).append(item)
                continue
            memory = self.accelerator.memory_of(target)
            if same := [s for s in item.sources if self.accelerator.memory_of(sources[s]) == memory]:
                pairs.add((same[0], item.target))
                delivered.append(item.box)
                continue
            if local := [s for s in item.sources if shares_memory(sources[s], target)]:
                pairs.add((local[0], item.target))
                in_place.setdefault(local[0], set()).add(target)
                continue
            distance = self._distances(self._anchor(target))
            s = min(item.sources, key=lambda s: (distance.get(self._anchor(sources[s]), float("inf")), sources[s].id))
            if self._anchor(sources[s]) not in distance:
                raise nx.NetworkXNoPath(f"no route from {sources[s]} to {target}")
            pairs.add((s, item.target))
            served.setdefault(s, []).append(item)

        load: dict[CommunicationLink, float] = {}
        order: dict[CommunicationLink, None] = {}
        ranked = sorted(served, key=lambda s: (-box_bits((i.box for i in served[s]), bitwidth), sources[s].id))
        for s in ranked:
            delivered += [i.box for i in served[s]]
            order |= dict.fromkeys(self._multicast(sources[s], targets, served[s], bitwidth, load))
        for s, reached in sorted(in_place.items()):
            for _, link in self._steiner_tree(sources[s], reached, 0.0, load, self._adjacency()):
                load.setdefault(link, 0.0)
                order[link] = None
        for t, items in sorted(reduced.items()):
            delivered += [i.box for i in items]
            order |= dict.fromkeys(self._reduce(sources, targets[t], items, bitwidth, load))
        moved = box_bits(delivered, bitwidth)
        shares = tuple((link, load[link] / moved if moved else 0.0) for link in order)
        return MulticastPathPlan(
            sources=sources,
            targets=targets,
            total_hops_objective=len(order),
            links_used=tuple(order),
            pairs=tuple(sorted(pairs)),
            link_shares=shares,
        )

    def _anchor(self, core: "Core") -> "Core":
        """Where data enters or leaves ``core``: the core owning its memory where that one has links, else itself."""
        owner = self.accelerator.memory_of(core)
        return owner if self._adjacency().get(owner) else core

    def _distances(self, target: "Core") -> dict["Core", float]:
        """Unloaded cycles per bit from every core to ``target``, cached per target."""
        cache = self.accelerator.__dict__.setdefault("_route_distances", {})
        if target not in cache:
            cost, _ = _dijkstra({target}, self._reverse_adjacency(), lambda link: 1.0 / link.bandwidth + _HOP)
            cache[target] = cost
        return cache[target]

    def _multicast(
        self,
        source: "Core",
        targets: tuple["Core", ...],
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
    ) -> list[CommunicationLink]:
        """Load ``load`` with, and return, the links of a tree from ``source`` to the targets of ``items``, each
        carrying the union of the data sent past it."""
        sent = box_bits((i.box for i in items), bitwidth)
        anchors = {self._anchor(targets[i.target]) for i in items}
        below: dict[CommunicationLink, set[Core]] = {}
        for (_, link), reached in self._steiner_tree(
            self._anchor(source), anchors, sent, load, self._adjacency()
        ).items():
            below.setdefault(link, set()).update(reached)
        for link, reached in below.items():
            bits = box_bits((i.box for i in items if self._anchor(targets[i.target]) in reached), bitwidth)
            load[link] = load.get(link, 0.0) + bits
        return list(below)

    def _reduce(
        self,
        sources: tuple["Core", ...],
        target: "Core",
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
    ) -> list[CommunicationLink]:
        """Load ``load`` with, and return, the links that bring the partial sums ``items`` from their sources to
        ``target``, adding them up on the way: a Steiner tree over the reversed links into the target, every link
        carrying the sum once. Partial sums in one memory are added there. Off-chip memory cannot add, so a sum headed
        there is completed in one of the on-chip memories next to it; cut into a part per memory holding partial sums,
        each completed where it then loads the busiest link least and all links least, the reduction becomes a
        reduce-scatter. A bus adds nothing, so each core sending over it sends its own sum."""
        reverse = self._reverse_adjacency()
        root = self._anchor(target)
        holders = [{self._anchor(sources[s]) for s in item.sources} - {root} for item in items]
        memories = set().union(*holders)
        if not memories:
            return []
        collectors = [(root, None)]
        parts = 1
        if root.type == "offchip":
            collectors = [(u, link) for u, link in reverse.get(root, ()) if u.type != "offchip"]
            parts = len(memories)
        used: dict[CommunicationLink, None] = {}
        for k in range(parts):
            boxes = [_part(item.box, parts, k) for item in items]
            bits = box_bits(boxes, bitwidth)
            if not bits:
                continue

            def plan(collector: "Core", last: CommunicationLink | None, bits: float = bits, boxes: list[Box] = boxes):
                below = self._steiner_tree(collector, memories - {collector}, bits, load, reverse)
                links = [
                    (link, box_bits((b for b, h in zip(boxes, holders, strict=True) if h & up), bitwidth))
                    for (_, link), up in below.items()
                ]
                return links + ([(last, bits)] if last is not None else [])

            options = [(plan(c, last), c.id) for c, last in collectors if memories <= self._distances(c).keys()]
            if not options:
                raise nx.NetworkXNoPath(f"no route to reduce into {target} from {sorted(m.id for m in memories)}")

            def busy(option: tuple[list[tuple[CommunicationLink, float]], int]) -> tuple[float, float, int, int]:
                cycles = [(load.get(link, 0.0) + b) / link.bandwidth for link, b in option[0]]
                return max(cycles), sum(cycles), len(cycles), option[1]

            links, _ = min(options, key=busy)
            for link, b in links:
                load[link] = load.get(link, 0.0) + b
                used[link] = None
        return list(used)

    def _steiner_tree(
        self,
        source: "Core",
        targets: set["Core"],
        bits: float,
        load: dict[CommunicationLink, float],
        adjacency: dict["Core", list[tuple["Core", CommunicationLink]]],
    ) -> dict[tuple["Core", CommunicationLink], set["Core"]]:
        """Per edge of a tree from ``source`` over ``adjacency`` spanning ``targets``, keyed by the core it leads to
        and its link, the targets reached through it."""
        parent: dict[Core, tuple[Core, CommunicationLink]] = {}
        tree = {source}
        tree_links: set[CommunicationLink] = set()
        remaining = set(targets) - tree

        def weight(link: CommunicationLink) -> float:
            if link in tree_links:
                return _HOP
            return (load.get(link, 0.0) + bits) / link.bandwidth + _HOP

        while remaining:
            cost, previous = _dijkstra(tree, adjacency, weight)
            reachable = [t for t in remaining if t in cost]
            if not reachable:
                raise nx.NetworkXNoPath(f"no route from {source} to {sorted(t.id for t in remaining)}")
            nearest = min(reachable, key=lambda t: (cost[t], t.id))
            node = nearest
            while node not in tree:
                u, link = previous[node]
                parent[node] = (u, link)
                tree.add(node)
                tree_links.add(link)
                node = u
            remaining -= tree
        below: dict[tuple[Core, CommunicationLink], set[Core]] = {}
        for target in targets:
            node = target
            while node != source and node in parent:
                u, link = parent[node]
                below.setdefault((node, link), set()).add(target)
                node = u
        return below


def _part(box: Box, parts: int, k: int) -> Box:
    """The ``k``-th of ``parts`` near-equal cuts of ``box`` along its longest axis."""
    if parts == 1:
        return box
    axis = max(range(len(box)), key=lambda a: box[a][1] - box[a][0])
    lo, hi = box[axis]
    edges = [lo + (hi - lo + 1) * j // parts for j in range(parts + 1)]
    return (*box[:axis], (edges[k], edges[k + 1] - 1), *box[axis + 1 :])


def _dijkstra(
    roots: set["Core"],
    adjacency: dict["Core", list[tuple["Core", CommunicationLink]]],
    weight: Callable[[CommunicationLink], float],
) -> tuple[dict["Core", float], dict["Core", tuple["Core", CommunicationLink]]]:
    """Cheapest cost from any of ``roots`` to every core, and the edge each is reached by; ties go to fewer hops, then
    the lower core id. Off-chip memory ends a route but never relays one: data between two cores does not pass
    through DRAM."""
    cost = {root: 0.0 for root in roots}
    previous: dict[Core, tuple[Core, CommunicationLink]] = {}
    heap = [(0.0, 0, root.id, root) for root in roots]
    heapq.heapify(heap)
    done: set[Core] = set()
    while heap:
        c, hops, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u.type == "offchip" and u not in roots:
            continue
        for v, link in adjacency.get(u, ()):
            nc = c + weight(link)
            if v not in done and nc < cost.get(v, float("inf")):
                cost[v] = nc
                previous[v] = (u, link)
                heapq.heappush(heap, (nc, hops + 1, v.id, v))
    return cost, previous
