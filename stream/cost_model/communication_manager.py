import heapq
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations, product
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


# One step of a route: the memory the data leaves, the one it reaches and the link it crosses.
Hop = tuple[Memory, Memory, CommunicationLink]


@dataclass(frozen=True, slots=True)
class MulticastPathPlan:
    """A transfer's route: which source serves which target (``pairs``, by position), the links the data crosses and
    the share of the transfer's bits each carries, one copy of the data per source however many targets it reaches.
    ``hop_demand`` holds the demand each of ``hops`` carries, ``delivered`` the boxes the transfer delivers."""

    sources: tuple[Memory, ...]
    targets: tuple[Memory, ...]
    total_hops_objective: int
    links_used: tuple["CommunicationLink", ...]
    pairs: tuple[tuple[int, int], ...] = ()
    link_shares: tuple[tuple["CommunicationLink", float], ...] = field(default=(), compare=False)
    hops: tuple[Hop, ...] = field(default=(), compare=False)
    hop_demand: tuple[tuple[DemandItem, ...], ...] = ()
    delivered: tuple[Box, ...] = field(default=(), compare=False)

    def links_at(self, memory: Memory, leaving: bool) -> tuple["CommunicationLink", ...]:
        """The links of the route's hops that leave ``memory`` (``leaving``) or enter it."""
        return tuple(dict.fromkeys(link for u, v, link in self.hops if (u if leaving else v) is memory))

    def share_at(self, memory: Memory, leaving: bool, links: Iterable["CommunicationLink"]) -> float:
        """The share of the delivered data ``memory`` sends over ``links`` (``leaving``) to the targets it serves, or
        receives over them for its own targets: all of a broadcast, the part of a split. Of data it passes on, it
        reads only what it holds as a target itself."""
        links, ends = set(links), self.sources if leaving else self.targets
        held = [
            item.box
            for (_, v, _), demand in zip(self.hops, self.hop_demand, strict=True)
            if leaving and v is memory
            for item in demand
            if self.targets[item.target] is memory
        ]
        boxes: list[Box] = []
        for (u, v, link), demand in zip(self.hops, self.hop_demand, strict=True):
            if link not in links or (u if leaving else v) is not memory:
                continue
            for item in demand:
                if any(ends[i] is memory for i in (item.sources if leaving else (item.target,))):
                    boxes.append(item.box)
                else:
                    boxes += [part for box in held if (part := meet(box, item.box))]
        delivered = box_bits(self.delivered, 1)
        return box_bits(boxes, 1) / delivered if delivered else 0.0

    def memory_share(self, memory: Memory, leaving: bool) -> float:
        """The share of the delivered data ``memory`` sends (``leaving``) or receives: all of it where it is the only
        memory on its side."""
        if set(self.sources if leaving else self.targets) == {memory}:
            return 1.0
        return self.share_at(memory, leaving, self.links_at(memory, leaving))


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
                links.sort(key=lambda edge: edge[0])
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
        """The routes a transfer between these memories that moves ``demand`` may take, cached per demand. The first is
        the fastest (:meth:`route` on idle links). Each of the up to ``alternatives - 1`` others is routed as if the
        earlier ones already carried the transfer, so it prefers other, comparably fast links. From off-chip memory to
        several targets the striped route is offered too. Routes whose hops carry the same data are dropped."""
        key = (tuple(src_allocs), tuple(dst_allocs), demand, bitwidth, alternatives)
        if key not in self._transfer_plans:
            found: dict[frozenset[tuple[Hop, frozenset[DemandItem]]], MulticastPathPlan] = {}
            taken: dict[CommunicationLink, float] = {}
            for _ in range(alternatives):
                plan, carried = self.route(key[0], key[1], demand, bitwidth, shares_memory, taken)
                found.setdefault(frozenset(zip(plan.hops, map(frozenset, plan.hop_demand), strict=True)), plan)
                for link, bits in carried.items():
                    taken[link] = taken.get(link, 0.0) + bits
            if len(set(key[1])) > 1 and any(source.offchip for source in key[0]):
                plan, _ = self.route(key[0], key[1], demand, bitwidth, shares_memory, striped=True)
                found.setdefault(frozenset(zip(plan.hops, map(frozenset, plan.hop_demand), strict=True)), plan)
            self._transfer_plans[key] = tuple(found.values())
        return self._transfer_plans[key]

    def route(  # noqa: PLR0913
        self,
        sources: tuple[Memory, ...],
        targets: tuple[Memory, ...],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[[Memory, Memory], bool],
        taken: Mapping[CommunicationLink, float] | None = None,
        striped: bool = False,
    ) -> tuple[MulticastPathPlan, dict[CommunicationLink, float]]:
        """A route of the transfer and the bits it puts on each link, given links already carrying ``taken`` bits
        (none by default).

        Each demand is served from the nearest source holding it. Then one Steiner tree per source grows over the
        targets it serves (the shortest-path heuristic of Takahashi and Matsuyama), largest source first, each link
        weighted by the cycles it would then be busy with the data it adds, so later trees avoid loaded links. A link
        carries the union of the data its tree sends past it. ``striped``, data from off-chip memory enters in parts
        (:meth:`_multicast`). A target that is its source needs no route. A target that shares memory with
        its source is reached by a route that moves no data. Partial sums are reduced on the way (:meth:`_reduce`).
        Each link's share is relative to all data the transfer delivers. ``taken`` only weighs the links."""
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
            s = min(item.sources, key=lambda s: (distance.get(sources[s], float("inf")), sources[s]))
            if sources[s] not in distance:
                raise nx.NetworkXNoPath(f"no route from {sources[s]} to {target}")
            pairs.add((s, item.target))
            served.setdefault(s, []).append(replace(item, sources=(s,)))

        taken = taken or {}
        load: dict[CommunicationLink, float] = dict(taken)
        hops: dict[Hop, dict[DemandItem, None]] = {}
        ranked = sorted(served, key=lambda s: (-box_bits((i.box for i in served[s]), bitwidth), sources[s]))
        for s in ranked:
            delivered += [i.box for i in served[s]]
            for hop, carried in self._multicast(sources[s], targets, served[s], bitwidth, load, striped).items():
                hops.setdefault(hop, {}).update(dict.fromkeys(carried))
        for s, reached in sorted(in_place.items()):
            for hop in self._steiner_tree(sources[s], dict.fromkeys(reached, ()), bitwidth, load, self._adjacency()):
                load.setdefault(hop[2], 0.0)
                hops.setdefault(hop, {})
        for t, items in sorted(reduced.items()):
            delivered += [i.box for i in items]
            for hop, carried in self._reduce(sources, targets[t], items, bitwidth, load).items():
                hops.setdefault(hop, {}).update(dict.fromkeys(carried))
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
            hop_demand=tuple(map(tuple, hops.values())),
            delivered=tuple(dict.fromkeys(delivered)),
        )
        return plan, carried

    def _distances(self, target: Memory) -> dict[Memory, float]:
        """Unloaded cycles per bit from every memory to ``target``, cached per target."""
        cache = self.accelerator.__dict__.setdefault("_route_distances", {})
        if target not in cache:
            cost, _ = _dijkstra({target}, self._reverse_adjacency(), lambda link: 1.0 / link.bandwidth + _HOP)
            cache[target] = cost
        return cache[target]

    def _multicast(  # noqa: PLR0913
        self,
        source: Memory,
        targets: tuple[Memory, ...],
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
        striped: bool = False,
    ) -> dict[Hop, list[DemandItem]]:
        """The hops of a tree from ``source`` to the targets of ``items``, with the items each carries, added to
        ``load``. ``striped`` from off-chip memory, the data every target it links to reads is cut into a part per such
        target, which enters over that target's link and takes the shortest paths on to the others."""
        adjacency = self._adjacency()
        read: dict[Memory, set[Box]] = {}
        for item in items:
            read.setdefault(targets[item.target], set()).add(item.box)
        entries = sorted({v for v, _ in adjacency[source]} & read.keys()) if striped and source.offchip else []
        common = set.intersection(*(read[entry] for entry in entries)) if len(entries) > 1 else set()
        rest = [item for item in items if item.box not in common]
        hops = self._tree(source, targets, rest, bitwidth, load, adjacency) if rest else {}
        for k, entry in enumerate(entries if common else ()):
            part = [replace(item, box=_part(item.box, len(entries), k)) for item in items if item.box in common]
            if not box_bits((item.box for item in part), bitwidth):
                continue
            through_entry = adjacency | {source: [(v, link) for v, link in adjacency[source] if v == entry]}
            for hop, carried in self._tree(source, targets, part, bitwidth, load, through_entry, spread=True).items():
                hops.setdefault(hop, []).extend(carried)
        return hops

    def _tree(  # noqa: PLR0913
        self,
        source: Memory,
        targets: tuple[Memory, ...],
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
        adjacency: dict[Memory, list[tuple[Memory, CommunicationLink]]],
        spread: bool = False,
    ) -> dict[Hop, list[DemandItem]]:
        """Add to ``load`` the links of a Steiner tree from ``source`` to the targets of ``items``, or of the shortest
        paths to them where ``spread``, each carrying the union of the data sent past it. Returns the items each hop
        carries."""
        needs: dict[Memory, tuple[Box, ...]] = {}
        for item in items:
            needs[targets[item.target]] = (*needs.get(targets[item.target], ()), item.box)
        grow = self._shortest_paths if spread else self._steiner_tree
        tree = grow(source, needs, bitwidth, load, adjacency)
        below: dict[CommunicationLink, set[Memory]] = {}
        for (_, _, link), reached in tree.items():
            below.setdefault(link, set()).update(reached)
        for link, reached in below.items():
            bits = box_bits((i.box for i in items if targets[i.target] in reached), bitwidth)
            load[link] = load.get(link, 0.0) + bits
        return {hop: [i for i in items if targets[i.target] in reached] for hop, reached in tree.items()}

    def _reduce(
        self,
        sources: tuple[Memory, ...],
        target: Memory,
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
    ) -> dict[Hop, list[DemandItem]]:
        """Add to ``load`` the links that bring the partial sums ``items`` from their sources to ``target``, adding
        them up on the way, and return the parts of ``items`` each of their hops carries: a Steiner tree over the
        reversed links into the target, every link carrying the sum once. Partial sums in one memory are added there.
        Off-chip memory cannot add, so a sum headed there is completed in one of the on-chip memories next to it; cut
        into a part per memory holding partial sums, each completed where it then loads the busiest link least and all
        links least, the reduction becomes a reduce-scatter. A bus adds nothing, so each memory sending over it sends
        its own sum."""
        reverse = self._reverse_adjacency()
        holders = [{sources[s] for s in item.sources} - {target} for item in items]
        memories = set().union(*holders)
        if not memories:
            return {}
        collectors = [(target, None)]
        parts = 1
        if target.offchip:
            collectors = [(u, link) for u, link in reverse.get(target, ()) if not u.offchip]
            parts = len(memories)
        used: dict[Hop, list[DemandItem]] = {}
        for k in range(parts):
            cut = [replace(item, box=_part(item.box, parts, k)) for item in items]
            bits = box_bits((item.box for item in cut), bitwidth)
            if not bits:
                continue

            def plan(collector: Memory, last: CommunicationLink | None, cut: list[DemandItem] = cut):
                # The tree grows over the reversed links, so a tree edge (u, v) is a hop of the sums from v to u.
                held = {m: tuple(i.box for i, h in zip(cut, holders, strict=True) if m in h) for m in memories}
                below = self._steiner_tree(collector, held, bitwidth, load, reverse)
                hops = [
                    ((v, u, link), [item for item, h in zip(cut, holders, strict=True) if h & up])
                    for (u, v, link), up in below.items()
                ]
                return hops + ([((collector, target, last), cut)] if last is not None else [])

            options = [(plan(c, last), c) for c, last in collectors if memories <= self._distances(c).keys()]
            if not options:
                raise nx.NetworkXNoPath(f"no route to reduce into {target} from {sorted(m.name for m in memories)}")

            def bits_of(carried: list[DemandItem]) -> float:
                return box_bits((item.box for item in carried), bitwidth)

            def busy(option: tuple[list[tuple[Hop, list[DemandItem]]], Memory]) -> tuple[float, float, int, Memory]:
                cycles = [
                    (load.get(link, 0.0) + bits_of(carried)) / link.bandwidth for (_, _, link), carried in option[0]
                ]
                return max(cycles), sum(cycles), len(cycles), option[1]

            hops, _ = min(options, key=busy)
            for hop, carried in hops:
                load[hop[2]] = load.get(hop[2], 0.0) + bits_of(carried)
                used.setdefault(hop, []).extend(carried)
        return used

    def _shortest_paths(
        self,
        source: Memory,
        needs: dict[Memory, tuple[Box, ...]],
        bitwidth: float,
        load: dict[CommunicationLink, float],
        adjacency: dict[Memory, list[tuple[Memory, CommunicationLink]]],
    ) -> dict[Hop, set[Memory]]:
        """Per hop of the shortest paths from ``source`` to the memories of ``needs``, the targets it leads to."""
        bits = box_bits({b for boxes in needs.values() for b in boxes}, bitwidth)

        def weight(link: CommunicationLink) -> float:
            return (load.get(link, 0.0) + bits) / link.bandwidth + _HOP

        cost, previous = _dijkstra({source}, adjacency, weight)
        if missing := [t for t in needs if t not in cost]:
            raise nx.NetworkXNoPath(f"no route from {source} to {sorted(t.name for t in missing)}")
        below: dict[Hop, set[Memory]] = {}
        for target in needs:
            node = target
            for u, link in _path(previous, target):
                below.setdefault((u, node, link), set()).add(target)
                node = u
        return below

    def _steiner_tree(
        self,
        source: Memory,
        needs: dict[Memory, tuple[Box, ...]],
        bitwidth: float,
        load: dict[CommunicationLink, float],
        adjacency: dict[Memory, list[tuple[Memory, CommunicationLink]]],
    ) -> dict[Hop, set[Memory]]:
        """Per hop of a tree from ``source`` over ``adjacency`` to the memories of ``needs``, the targets reached
        through it. A link is weighted by the cycles it would be busy with the data a target adds to what it already
        carries, a hop where it carries all of it. Targets reading as much data that no link carries yet, and the same
        data some link carries, share a search, redone only where a link of its path took more data since."""
        bits, cells = _cells(needs, bitwidth)
        held: dict[CommunicationLink, set[tuple]] = {}
        held_bits: dict[CommunicationLink, float] = {}
        sent: set[tuple] = set()
        below: dict[Hop, set[Memory]] = {}
        remaining = set(needs) - {source}
        searches: dict[tuple, tuple[dict[Memory, float], dict[Memory, tuple[Memory, CommunicationLink]]]] = {}
        changed: dict[tuple, set[CommunicationLink]] = {}

        def group(target: Memory) -> tuple[float, frozenset[tuple]]:
            return sum(bits[c] for c in cells[target] - sent), cells[target] & sent

        def added(link: CommunicationLink, key: tuple[float, frozenset[tuple]]) -> float:
            fresh, seen = key
            taken = held.get(link, set())
            return fresh + sum(bits[c] for c in seen if c not in taken)

        def search(key: tuple[float, frozenset[tuple]]) -> None:
            def weight(link: CommunicationLink) -> float:
                if not (extra := added(link, key)):
                    return _HOP
                return (load.get(link, 0.0) + held_bits.get(link, 0.0) + extra) / link.bandwidth + _HOP

            searches[key] = _dijkstra({source}, adjacency, weight)
            changed[key] = set()

        while remaining:
            while True:
                for key in {group(t) for t in remaining} - searches.keys():
                    search(key)
                reachable = [t for t in remaining if t in searches[group(t)][0]]
                if not reachable:
                    raise nx.NetworkXNoPath(f"no route from {source} to {sorted(t.name for t in remaining)}")
                target = min(reachable, key=lambda t: (searches[group(t)][0][t], t))
                steps = _path(searches[group(target)][1], target)
                if not changed[group(target)] & {link for _, link in steps}:
                    break
                del searches[group(target)]
            key, node = group(target), target
            for u, link in steps:
                below.setdefault((u, node, link), set()).add(target)
                held_bits[link] = held_bits.get(link, 0.0) + added(link, key)
                held.setdefault(link, set()).update(cells[target])
                for links in changed.values():
                    links.add(link)
                node = u
            sent |= cells[target]
            remaining.discard(target)
            for stale in searches.keys() - {group(t) for t in remaining}:
                del searches[stale], changed[stale]
        return below


def _path(
    previous: dict[Memory, tuple[Memory, CommunicationLink]], target: Memory
) -> list[tuple[Memory, CommunicationLink]]:
    """The edges a search reached ``target`` by, from it back to the search's root."""
    steps, node = [], target
    while node in previous:
        node, link = previous[node]
        steps.append((node, link))
    return steps


def meet(a: Box, b: Box) -> Box | None:
    """The part two boxes share, if any."""
    if _disjoint(a, b):
        return None
    return tuple((max(la, lb), min(ha, hb)) for (la, ha), (lb, hb) in zip(a, b, strict=True))


def _cells(
    needs: dict[Memory, tuple[Box, ...]], bitwidth: float
) -> tuple[dict[tuple, float], dict[Memory, frozenset[tuple]]]:
    """The data of ``needs`` cut into cells where any memory's boxes begin or end: the bits of each cell, by its first
    corner, and the cells each memory reads."""
    boxes = {b for needed in needs.values() for b in needed}
    axes = len(next(iter(boxes))) if boxes else 0
    edges = [sorted({b[a][0] for b in boxes} | {b[a][1] + 1 for b in boxes}) for a in range(axes)]
    bits: dict[tuple, float] = {}
    cells: dict[Memory, frozenset[tuple]] = {}
    for memory, needed in needs.items():
        read: set[tuple] = set()
        for box in needed:
            spans = []
            for a, (lo, hi) in enumerate(box):
                cuts = edges[a][bisect_left(edges[a], lo) : bisect_right(edges[a], hi + 1)]
                spans.append([(c, d - c) for c, d in zip(cuts, cuts[1:], strict=False)])
            for cell in product(*spans):
                corner = tuple(c for c, _ in cell)
                bits[corner] = prod(n for _, n in cell) * bitwidth
                read.add(corner)
        cells[memory] = frozenset(read)
    return bits, cells


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
    then the memory that sorts first. Off-chip memory ends a route but never relays one: data between two memories does
    not pass through DRAM."""
    cost = {root: 0.0 for root in roots}
    previous: dict[Memory, tuple[Memory, CommunicationLink]] = {}
    heap = [(0.0, 0, root) for root in roots]
    heapq.heapify(heap)
    done: set[Memory] = set()
    while heap:
        c, hops, u = heapq.heappop(heap)
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
                heapq.heappush(heap, (nc, hops + 1, v))
    return cost, previous
