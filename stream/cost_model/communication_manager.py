import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import combinations
from math import prod
from typing import TYPE_CHECKING

import networkx as nx

from stream.hardware.architecture.memory import Memory
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

    sources: tuple[Memory, ...]
    targets: tuple[Memory, ...]
    total_hops_objective: int
    links_used: tuple["CommunicationLink", ...]
    pairs: tuple[tuple[int, int], ...] = ()
    link_shares: tuple[tuple["CommunicationLink", float], ...] = field(default=(), compare=False)
    hops: tuple["Hop", ...] = field(default=(), compare=False)
    """Each step the data takes, (from memory, to memory, over link); a bus is one link but as many hops as memories
    it serves."""

    def links_at(self, memory: Memory, leaving: bool) -> tuple["CommunicationLink", ...]:
        """The links the route's data leaves ``memory`` over, or enters it over: those of the hops out of it, or into
        it (a bus only where a hop of the route ends at the memory, not for every member on it). A plan without hops,
        as a test builds, has links out of or into it and the buses it is on."""
        if self.hops:
            return tuple(dict.fromkeys(link for u, v, link in self.hops if (u if leaving else v) is memory))
        return tuple(
            link
            for link in self.links_used
            if memory in link.members or (link.sender if leaving else link.receiver) is memory
        )


Hop = tuple[Memory, Memory, "CommunicationLink"]
"""One step of a route: the memory the data leaves, the one it reaches and the link it crosses."""


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

    def __init__(self, accelerator: "Accelerator") -> None:
        self.accelerator = accelerator
        self._transfer_plans: dict[tuple, tuple[MulticastPathPlan, ...]] = {}

    def _adjacency(self) -> dict[Memory, list[tuple[Memory, CommunicationLink]]]:
        """Each memory's outgoing links, computed once per accelerator."""
        adjacency = self.accelerator.__dict__.get("_route_adjacency")
        if adjacency is None:
            adjacency = {memory: [] for memory in self.accelerator.memories.values()}
            for u, v, data in self.accelerator.network.edges(data=True):
                adjacency[u].append((v, data["cl"]))
            for links in adjacency.values():
                links.sort(key=lambda edge: edge[0].name)
            self.accelerator.__dict__["_route_adjacency"] = adjacency
        return adjacency

    def _reverse_adjacency(self) -> dict[Memory, list[tuple[Memory, CommunicationLink]]]:
        """Each memory's incoming links, computed once per accelerator."""
        reverse = self.accelerator.__dict__.get("_route_reverse_adjacency")
        if reverse is None:
            reverse = {memory: [] for memory in self.accelerator.memories.values()}
            for u, edges in self._adjacency().items():
                for v, link in edges:
                    reverse[v].append((u, link))
            self.accelerator.__dict__["_route_reverse_adjacency"] = reverse
        return reverse

    def get_possible_transfer_plan(
        self,
        src_allocs: Iterable[Memory],
        dst_allocs: Iterable[Memory],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[[Memory, Memory], bool],
        alternatives: int,
    ) -> tuple[MulticastPathPlan, ...]:
        """The routes a transfer between these memories that moves ``demand`` may take, planned once per demand: the
        fastest (:meth:`route`), then up to ``alternatives - 1`` more, each routed as if the ones before it already
        carried the transfer, so that it takes other connections where comparably fast ones exist and the solve can
        spread transfers sharing a slot over them. A route over the same links as an earlier one is dropped."""
        if alternatives < 1:
            raise ValueError(f"a transfer takes one of at least one route, not {alternatives}")
        key = (tuple(src_allocs), tuple(dst_allocs), demand, bitwidth, alternatives)
        if (plans := self._transfer_plans.get(key)) is None:
            found: dict[frozenset[CommunicationLink], MulticastPathPlan] = {}
            taken: dict[CommunicationLink, float] = {}
            for _ in range(alternatives):
                plan, carried = self._route(key[0], key[1], demand, bitwidth, shares_memory, taken)
                found.setdefault(frozenset(plan.links_used), plan)
                for link, bits in carried.items():
                    taken[link] = taken.get(link, 0.0) + bits
            plans = self._transfer_plans[key] = tuple(found.values())
        return plans

    def route(
        self,
        sources: tuple[Memory, ...],
        targets: tuple[Memory, ...],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[[Memory, Memory], bool],
    ) -> MulticastPathPlan:
        """The fastest route of a transfer, see :meth:`_route`."""
        plan, _ = self._route(sources, targets, demand, bitwidth, shares_memory, {})
        return plan

    def _route(
        self,
        sources: tuple[Memory, ...],
        targets: tuple[Memory, ...],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[[Memory, Memory], bool],
        taken: dict[CommunicationLink, float],
    ) -> tuple[MulticastPathPlan, dict[CommunicationLink, float]]:
        """A route of the transfer, and the bits it puts on each link, with the links already carrying ``taken`` bits.
        Serve each demand from the nearest source holding it, then grow one Steiner tree per source over the
        targets it serves (the shortest-path heuristic of Takahashi and Matsuyama), the largest source first and each
        link weighted by the cycles it would then be busy, so later trees avoid what earlier ones load. A link carries
        the union of the data its tree sends past it. A target that is its source needs no route; one that reaches its
        source otherwise, as a neighbouring tile's memory does, is reached on a route that moves none of it. Partial
        sums are then reduced on the way, see :meth:`_reduce`. Each link's share is of all the data the transfer
        delivers, whether it moves or is already in its target; what ``taken`` carries only weighs the links."""
        served: dict[int, list[DemandItem]] = {}
        in_place: dict[int, set[Memory]] = {}
        pairs: set[tuple[int, int]] = set()
        reduced: dict[int, list[DemandItem]] = {}
        delivered: list[Box] = []
        for item in demand:
            target = targets[item.target]
            if item.reduce:
                pairs.update((s, item.target) for s in item.sources)
                reduced.setdefault(item.target, []).append(item)
                continue
            if same := [s for s in item.sources if sources[s] is target]:
                pairs.add((same[0], item.target))
                delivered.append(item.box)
                continue
            if local := [s for s in item.sources if shares_memory(sources[s], target)]:
                pairs.add((local[0], item.target))
                in_place.setdefault(local[0], set()).add(target)
                continue
            distance = self._distances(target)
            s = min(item.sources, key=lambda s: (distance.get(sources[s], float("inf")), sources[s].name))
            if sources[s] not in distance:
                raise nx.NetworkXNoPath(f"no route from {sources[s]} to {target}")
            pairs.add((s, item.target))
            served.setdefault(s, []).append(item)

        load: dict[CommunicationLink, float] = dict(taken)
        hops: dict[Hop, None] = {}
        ranked = sorted(served, key=lambda s: (-box_bits((i.box for i in served[s]), bitwidth), sources[s].name))
        for s in ranked:
            delivered += [i.box for i in served[s]]
            hops |= dict.fromkeys(self._multicast(sources[s], targets, served[s], bitwidth, load))
        for s, reached in sorted(in_place.items()):
            for hop in self._steiner_tree(sources[s], reached, 0.0, load, self._adjacency()):
                load.setdefault(hop[2], 0.0)
                hops[hop] = None
        for t, items in sorted(reduced.items()):
            delivered += [i.box for i in items]
            hops |= dict.fromkeys(self._reduce(sources, targets[t], items, bitwidth, load))
        links = dict.fromkeys(link for _, _, link in hops)
        moved = box_bits(delivered, bitwidth)
        carried = {link: load[link] - taken.get(link, 0.0) for link in links}
        shares = tuple((link, carried[link] / moved if moved else 0.0) for link in links)
        plan = MulticastPathPlan(
            sources=sources,
            targets=targets,
            total_hops_objective=len(links),
            links_used=tuple(links),
            pairs=tuple(sorted(pairs)),
            link_shares=shares,
            hops=tuple(hops),
        )
        return plan, carried

    def _distances(self, target: Memory) -> dict[Memory, float]:
        """Unloaded cycles per bit from every memory to ``target``, cached per target."""
        cache = self.accelerator.__dict__.setdefault("_route_distances", {})
        if target not in cache:
            cost, _ = _dijkstra({target}, self._reverse_adjacency(), lambda link: 1.0 / link.bandwidth + _HOP)
            cache[target] = cost
        return cache[target]

    def _multicast(
        self,
        source: Memory,
        targets: tuple[Memory, ...],
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
    ) -> list[Hop]:
        """Load ``load`` with the links of a tree from ``source`` to the targets of ``items``, each carrying the union
        of the data sent past it, and return the tree's hops."""
        sent = box_bits((i.box for i in items), bitwidth)
        ends = {targets[i.target] for i in items}
        tree = self._steiner_tree(source, ends, sent, load, self._adjacency())
        below: dict[CommunicationLink, set[Memory]] = {}
        for (_, _, link), reached in tree.items():
            below.setdefault(link, set()).update(reached)
        for link, reached in below.items():
            bits = box_bits((i.box for i in items if targets[i.target] in reached), bitwidth)
            load[link] = load.get(link, 0.0) + bits
        return list(tree)

    def _reduce(
        self,
        sources: tuple[Memory, ...],
        target: Memory,
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
    ) -> list[Hop]:
        """Load ``load`` with the links that bring the partial sums ``items`` from their sources to ``target``, adding
        them up on the way, and return their hops: a Steiner tree over the reversed links into the target, every link
        carrying the sum once. Partial sums in one memory are added there. Off-chip memory cannot add, so a sum headed
        there is completed in one of the on-chip memories next to it; cut into a part per memory holding partial sums,
        each completed where it then loads the busiest link least and all links least, the reduction becomes a
        reduce-scatter. A bus adds nothing, so each memory sending over it sends its own sum."""
        reverse = self._reverse_adjacency()
        root = target
        holders = [{sources[s] for s in item.sources} - {root} for item in items]
        memories = set().union(*holders)
        if not memories:
            return []
        collectors = [(root, None)]
        parts = 1
        if root.offchip:
            collectors = [(u, link) for u, link in reverse.get(root, ()) if not u.offchip]
            parts = len(memories)
        used: dict[Hop, None] = {}
        for k in range(parts):
            boxes = [_part(item.box, parts, k) for item in items]
            bits = box_bits(boxes, bitwidth)
            if not bits:
                continue

            def plan(collector: Memory, last: CommunicationLink | None, bits: float = bits, boxes: list[Box] = boxes):
                # The tree grows over the reversed links, so a tree edge (u, v) is a hop of the sums from v to u.
                below = self._steiner_tree(collector, memories - {collector}, bits, load, reverse)
                hops = [
                    ((v, u, link), box_bits((b for b, h in zip(boxes, holders, strict=True) if h & up), bitwidth))
                    for (u, v, link), up in below.items()
                ]
                return hops + ([((collector, root, last), bits)] if last is not None else [])

            options = [(plan(c, last), c.name) for c, last in collectors if memories <= self._distances(c).keys()]
            if not options:
                raise nx.NetworkXNoPath(f"no route to reduce into {target} from {sorted(m.name for m in memories)}")

            def busy(option: tuple[list[tuple[Hop, float]], str]) -> tuple[float, float, int, str]:
                cycles = [(load.get(link, 0.0) + b) / link.bandwidth for (_, _, link), b in option[0]]
                return max(cycles), sum(cycles), len(cycles), option[1]

            hops, _ = min(options, key=busy)
            for hop, b in hops:
                load[hop[2]] = load.get(hop[2], 0.0) + b
                used[hop] = None
        return list(used)

    def _steiner_tree(
        self,
        source: Memory,
        targets: set[Memory],
        bits: float,
        load: dict[CommunicationLink, float],
        adjacency: dict[Memory, list[tuple[Memory, CommunicationLink]]],
    ) -> dict[Hop, set[Memory]]:
        """Per edge of a tree from ``source`` over ``adjacency`` spanning ``targets``, keyed by the memory it leaves,
        the one it leads to and its link, the targets reached through it."""
        parent: dict[Memory, tuple[Memory, CommunicationLink]] = {}
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
                raise nx.NetworkXNoPath(f"no route from {source} to {sorted(t.name for t in remaining)}")
            nearest = min(reachable, key=lambda t: (cost[t], t.name))
            node = nearest
            while node not in tree:
                u, link = previous[node]
                parent[node] = (u, link)
                tree.add(node)
                tree_links.add(link)
                node = u
            remaining -= tree
        below: dict[Hop, set[Memory]] = {}
        for target in targets:
            node = target
            while node != source and node in parent:
                u, link = parent[node]
                below.setdefault((u, node, link), set()).add(target)
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
    roots: set[Memory],
    adjacency: dict[Memory, list[tuple[Memory, CommunicationLink]]],
    weight: Callable[[CommunicationLink], float],
) -> tuple[dict[Memory, float], dict[Memory, tuple[Memory, CommunicationLink]]]:
    """Cheapest cost from any of ``roots`` to every memory, and the edge each is reached by; ties go to fewer hops,
    then the lower memory name. Off-chip memory ends a route but never relays one: data between two memories does not
    pass through DRAM."""
    cost = {root: 0.0 for root in roots}
    previous: dict[Memory, tuple[Memory, CommunicationLink]] = {}
    heap = [(0.0, 0, root.name, root) for root in roots]
    heapq.heapify(heap)
    done: set[Memory] = set()
    while heap:
        c, hops, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u.offchip and u not in roots:
            continue
        for v, link in adjacency.get(u, ()):
            nc = c + weight(link)
            if v not in done and nc < cost.get(v, float("inf")):
                cost[v] = nc
                previous[v] = (u, link)
                heapq.heappush(heap, (nc, hops + 1, v.name, v))
    return cost, previous
