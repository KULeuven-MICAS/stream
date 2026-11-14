import heapq
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from itertools import product
from math import prod
from typing import TYPE_CHECKING, Literal

import networkx as nx

from stream.hardware.architecture.memory import Memory
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.hardware.ports import READ
from stream.workload.demand import Box, DemandItem, box_bits, covered, firing_count, meet, moved_elements

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.mapping.mapping import EdgePlacement

_HOP = 1e-9
"""Cost of a hop beside the cycles a link adds, so equally fast routes prefer the shorter."""


# One step of a route: the memory the data leaves, the one it reaches and the link it crosses.
Hop = tuple[Memory, Memory, CommunicationLink]

# What a memory sends of each item crossing a link out of it, each part with the copies it holds that part for.
Sent = list[tuple[CommunicationLink, DemandItem, list[tuple[tuple[int, ...], Box]]]]

# How a route leaves off-chip memory: as one tree, in a part per linked target, or as a tree over one of its links.
Entry = Literal["tree", "striped", "single"]


@dataclass(frozen=True, slots=True)
class MulticastPathPlan:
    """A transfer's route: the source serving each target (``pairs``), the ``hops`` it crosses and the share of its bits
    each link carries, one copy per source; ``hop_demand`` is the demand on each hop, ``delivered`` the boxes it
    delivers, ``items`` the demand it delivers, ``demand`` all its targets read, ``placement`` how its sources hold an
    input whose placement the allocation chooses. Data a target already holds is not delivered."""

    sources: tuple[Memory, ...]
    targets: tuple[Memory, ...]
    total_hops_objective: int
    links_used: tuple["CommunicationLink", ...]
    pairs: tuple[tuple[int, int], ...] = ()
    link_shares: tuple[tuple["CommunicationLink", float], ...] = field(default=(), compare=False)
    hops: tuple[Hop, ...] = field(default=(), compare=False)
    hop_demand: tuple[tuple[DemandItem, ...], ...] = ()
    delivered: tuple[Box, ...] = field(default=(), compare=False)
    delivered_items: tuple[DemandItem, ...] = field(default=(), compare=False)
    demand: tuple[DemandItem, ...] = field(default=(), compare=False)
    placement: "EdgePlacement | None" = None
    _shares: dict[tuple, float] = field(default_factory=dict, init=False, compare=False, repr=False)
    _sent: dict[Memory, Sent] = field(default_factory=dict, init=False, compare=False, repr=False)
    _hash: list[int] = field(default_factory=list, init=False, compare=False, repr=False)

    def __hash__(self) -> int:
        """The hash of the fields compared, computed once: a route keys most of what a model builds per route."""
        if not self._hash:
            compared = (self.sources, self.targets, self.total_hops_objective, self.links_used, self.pairs)
            self._hash.append(hash((*compared, self.hop_demand, self.placement)))
        return self._hash[0]

    def links_at(self, memory: Memory, leaving: bool) -> tuple["CommunicationLink", ...]:
        """The links of the route's hops that leave ``memory`` (``leaving``) or enter it."""
        return tuple(dict.fromkeys(link for u, v, link in self.hops if (u if leaving else v) is memory))

    def share_at(
        self, memory: Memory, leaving: bool, links: Iterable["CommunicationLink"], copies: tuple[int, ...] | None = None
    ) -> float:
        """The share of the delivered data ``memory`` sends over ``links`` (``leaving``) to the targets it serves, or
        receives over them for its own targets: all of a broadcast, the part of a split. Of data it passes on, it
        reads only what it holds as a target itself. Off-chip memory, with no switch, sends a copy per link.
        With ``copies``, the share of what it delivers to exactly those copies. Partial sums count wherever they
        arrive or leave."""
        links = frozenset(links)
        key = (memory, leaving, links, copies)
        if (found := self._shares.get(key)) is None:
            found = self._shares[key] = self._share_at(memory, leaving, links, copies)
        return found

    def _share_at(
        self, memory: Memory, leaving: bool, links: frozenset["CommunicationLink"], copies: tuple[int, ...] | None
    ) -> float:
        """:meth:`share_at`, uncached."""
        kept = _kept(copies)
        boxes: dict[CommunicationLink | int | None, list[Box]] = {}
        if leaving:
            for link, item, parts in self._sent_from(memory):
                if link in links and kept(item):
                    own = boxes.setdefault(link if memory.offchip else None, [])
                    own += [part for held, part in parts if copies is None or held == copies]
        else:
            for k, ((_, v, link), demand) in enumerate(zip(self.hops, self.hop_demand, strict=True)):
                if link in links and v is memory:
                    own = boxes.setdefault(None, [])
                    own += [i.box for i in demand if kept(i) and not i.reduce and self.targets[i.target] is memory]
                    boxes[k] = [item.box for item in demand if kept(item) and item.reduce]
        return self._of_delivered(sum(box_bits(own, 1) for own in boxes.values()), copies)

    def _of_delivered(self, points: float, copies: tuple[int, ...] | None) -> float:
        """``points`` of all firings' boxes as a share of the delivered data, or of what is delivered to ``copies``."""
        if (delivered := self._shares.get(("delivered", copies))) is None:
            delivered = self._shares["delivered", copies] = moved_elements(filter(_kept(copies), self.delivered_items))
        return points / firing_count(self.demand) / delivered if delivered else 0.0

    def sum_read_share(
        self, memory: Memory, links: Iterable["CommunicationLink"], copies: tuple[int, ...] | None = None
    ) -> float:
        """The share of the delivered data (of ``copies``) read from ``memory`` to add the partial sums arriving over
        ``links`` into a sum held there: its own or one received before."""
        links = frozenset(links)
        key = ("sum_read", memory, links, copies)
        if (found := self._shares.get(key)) is not None:
            return found
        kept = _kept(copies)
        mine = [i.box for i in self.delivered_items if i.reduce and kept(i) and self._sources_in(i, memory)]
        added = 0
        for (_, v, link), demand in zip(self.hops, self.hop_demand, strict=True):
            if v is memory:
                arriving = [item.box for item in demand if item.reduce and kept(item)]
                if link in links:
                    added += covered(part for box in arriving for held in mine if (part := meet(box, held)))
                mine += arriving
        found = self._shares[key] = self._of_delivered(added, copies)
        return found

    def local_sum_share(
        self, memory: Memory, links: Iterable["CommunicationLink"], copies: tuple[int, ...] | None = None
    ) -> float:
        """The share of the delivered data (of ``copies``) read from ``memory``, and written to it as their sum, to add
        up the partial sums of one box that several sources there hold: the box once per source past the first, on the
        links the sum leaves on, or where it is collected there, the first it arrives on."""
        links = frozenset(links)
        key = ("local_sum", memory, links, copies)
        if (found := self._shares.get(key)) is not None:
            return found
        kept = _kept(copies)
        points = 0
        for item in self.delivered_items:
            if not (item.reduce and kept(item)) or (extra := self._sources_in(item, memory) - 1) < 1:
                continue
            on = (
                self._partial_sum_links(memory, item, leaving=True)
                or self._partial_sum_links(memory, item, leaving=False)[:1]
            )
            points += extra * covered(part for link, part in on if link in links)
        found = self._shares[key] = self._of_delivered(points, copies)
        return found

    def _sources_in(self, item: DemandItem, memory: Memory) -> int:
        """How many of ``item``'s sources are in ``memory``."""
        return sum(self.sources[s] is memory for s in item.sources)

    def _partial_sum_links(
        self, memory: Memory, item: DemandItem, leaving: bool
    ) -> list[tuple["CommunicationLink", Box]]:
        """The links of the hops that carry partial sums of ``item`` out of ``memory`` (``leaving``) or into it, each
        with the part of the box it carries."""
        return [
            (link, part)
            for (u, v, link), demand in zip(self.hops, self.hop_demand, strict=True)
            if (u if leaving else v) is memory
            for carried in demand
            if carried.reduce and (part := meet(carried.box, item.box))
        ]

    def sum_memories(self) -> tuple[Memory, ...]:
        """The memories a partial sum is added into by the core owning them: those partial sums arrive in, and those
        holding several of one box."""
        arriving = (
            v for (_, v, _), demand in zip(self.hops, self.hop_demand, strict=True) if any(i.reduce for i in demand)
        )
        holding = (m for i in self.delivered_items if i.reduce for m in self.sources if self._sources_in(i, m) > 1)
        return tuple(dict.fromkeys((*arriving, *holding)))

    def _sent_from(self, memory: Memory) -> Sent:
        """Per item ``memory`` sends over a link of the route, the parts it sends with the copies they are for: all of
        the item where it is a source, else what it holds of it as a target."""
        if (found := self._sent.get(memory)) is not None:
            return found
        held = dict.fromkeys(
            (item.copies, item.box)
            for (_, v, _), demand in zip(self.hops, self.hop_demand, strict=True)
            if v is memory
            for item in demand
            if self.targets[item.target] is memory
        )
        parts: dict[Box, list[tuple[tuple[int, ...], Box]]] = {}
        found = self._sent[memory] = []
        for (u, _, link), demand in zip(self.hops, self.hop_demand, strict=True):
            for item in demand if u is memory else ():
                if item.reduce or any(self.sources[i] is memory for i in item.sources):
                    found.append((link, item, [(item.copies, item.box)]))
                    continue
                if (passed := parts.get(item.box)) is None:
                    passed = parts[item.box] = [(c, part) for c, box in held if (part := meet(box, item.box))]
                found.append((link, item, passed))
        return found

    def unique_sent_share(self, memory: Memory, copies: tuple[int, ...] | None = None) -> float:
        """The share of the delivered data (of ``copies``) ``memory`` sends, each element once whatever the links."""
        key = ("unique_sent", memory, copies)
        if (found := self._shares.get(key)) is None:
            kept = _kept(copies)
            sent = [
                p for _, item, parts in self._sent_from(memory) if kept(item) for c, p in parts if copies in (None, c)
            ]
            found = self._shares[key] = self._of_delivered(covered(sent), copies)
        return found

    def memory_share(self, memory: Memory, leaving: bool, copies: tuple[int, ...] | None = None) -> float:
        """The share of the delivered data ``memory`` sends (``leaving``) or receives: all of it where it is the only
        memory on its side, but for off-chip memory sending, a copy per link (:meth:`share_at`)."""
        links = self.links_at(memory, leaving)
        alone = set(self.sources if leaving else self.targets) == {memory} and not any(
            i.reduce for i in self.delivered_items
        )
        if alone and not (leaving and memory.offchip and links):
            return 1.0
        return self.share_at(memory, leaving, links, copies)


def _kept(copies: tuple[int, ...] | None) -> Callable[[DemandItem], bool]:
    """Whether an item is delivered to exactly ``copies``; every item where None."""
    return (lambda item: item.copies == copies) if copies is not None else (lambda item: True)


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
        """The routes a transfer moving ``demand`` between these memories may take, cached per demand: the fastest
        first, then up to ``alternatives - 1`` routed as if the earlier ones carried it, and from off-chip memory to
        several targets the striped route and the tree entering over one link, where one exists. Routes whose hops
        carry the same data are dropped."""
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
                single: tuple[Entry, ...] = () if self._enters_each(key[0], key[1]) else ("single",)
                for entry in ("striped", *single):
                    with suppress(nx.NetworkXNoPath):
                        plan, _ = self.route(key[0], key[1], demand, bitwidth, shares_memory, entry=entry)
                        found.setdefault(frozenset(zip(plan.hops, map(frozenset, plan.hop_demand), strict=True)), plan)
            self._transfer_plans[key] = tuple(found.values())
        return self._transfer_plans[key]

    def _enters_each(self, sources: tuple[Memory, ...], targets: tuple[Memory, ...]) -> bool:
        """Whether every target has a link of its own from an off-chip source that carries more than one transfer at
        once, so the striped route needs no shared entry and the route over one link is never the faster."""
        from stream.opt.allocation.constraint_optimization.families.routing import circuits  # noqa: PLC0415

        adjacency = self._adjacency()
        for target in set(targets):
            direct = [link for source in sources if source.offchip for v, link in adjacency[source] if v is target]
            if not direct or not any(circuits(link) > 1 for link in direct):
                return False
        return True

    def route(  # noqa: PLR0913
        self,
        sources: tuple[Memory, ...],
        targets: tuple[Memory, ...],
        demand: tuple[DemandItem, ...],
        bitwidth: float,
        shares_memory: Callable[[Memory, Memory], bool],
        taken: Mapping[CommunicationLink, float] | None = None,
        entry: Entry = "tree",
    ) -> tuple[MulticastPathPlan, dict[CommunicationLink, float]]:
        """A route for the transfer and each link's share of the bits it delivers. Each source grows a Steiner tree over
        its targets, weighting a link by the cycles its added data keeps it busy beside ``taken``. Off-chip data enters
        in parts if ``striped``, over one link if ``single``."""
        served: dict[int, list[DemandItem]] = {}
        in_place: dict[int, set[Memory]] = {}
        pairs: set[tuple[int, int]] = set()
        reduced: dict[int, list[DemandItem]] = {}
        delivered: list[Box] = []
        routed: list[DemandItem] = []
        for item in demand:
            target = targets[item.target]
            if not item.sources:
                continue
            if item.reduce:
                pairs.update((s, item.target) for s in item.sources)
                reduced.setdefault(item.target, []).append(item)
                continue
            if same := [s for s in item.sources if sources[s] is target]:
                pairs.add((same[0], item.target))
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

        taken, spanned = taken or {}, firing_count(demand)
        load: dict[CommunicationLink, float] = {link: bits * spanned for link, bits in taken.items()}
        hops: dict[Hop, dict[DemandItem, None]] = {}
        ranked = sorted(served, key=lambda s: (-box_bits((i.box for i in served[s]), bitwidth), sources[s]))
        for s in ranked:
            delivered += [i.box for i in served[s]]
            routed += served[s]
            for hop, carried in self._multicast(sources[s], targets, served[s], bitwidth, load, entry).items():
                hops.setdefault(hop, {}).update(dict.fromkeys(carried))
        for s, reached in sorted(in_place.items()):
            for hop in self._steiner_tree(sources[s], dict.fromkeys(reached, ()), bitwidth, load, self._adjacency()):
                load.setdefault(hop[2], 0.0)
                hops.setdefault(hop, {})
        for t, items in sorted(reduced.items()):
            delivered += [i.box for i in items]
            routed += items
            for hop, carried in self._reduce(sources, targets[t], items, bitwidth, load).items():
                hops.setdefault(hop, {}).update(dict.fromkeys(carried))
        links = dict.fromkeys(link for _, _, link in hops)
        moved = moved_elements(routed) * bitwidth
        carried = {link: load[link] / spanned - taken.get(link, 0.0) for link in links}
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
            delivered_items=tuple(routed),
            demand=demand,
        )
        return plan, carried

    def _distances(self, target: Memory) -> dict[Memory, float]:
        """Unloaded cycles per bit from every memory to ``target``, cached per target."""
        cache = self.accelerator.__dict__.setdefault("_route_distances", {})
        if target not in cache:
            cost, _ = _dijkstra({target}, self._reverse_adjacency(), lambda _, link: 1.0 / link.bandwidth + _HOP)
            cache[target] = cost
        return cache[target]

    def _multicast(  # noqa: PLR0913
        self,
        source: Memory,
        targets: tuple[Memory, ...],
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
        entry: Entry = "tree",
    ) -> dict[Hop, list[DemandItem]]:
        """The hops of a tree from ``source`` to the targets of ``items``, with the items each carries, added to
        ``load``. ``striped`` from off-chip memory, what all the targets it has a link (not a bus) to read enters a part
        over each; ``single``, all of it over the one link out of ``source`` that keeps the tree least busy."""
        adjacency = self._adjacency()
        direct = [(v, link) for v, link in adjacency[source] if not link.members]
        if entry == "single" and direct:
            return self._single_entry(source, targets, items, bitwidth, load, direct)
        read: dict[Memory, set[Box]] = {}
        for item in items:
            read.setdefault(targets[item.target], set()).add(item.box)
        entries = sorted({v for v, _ in direct} & read.keys()) if entry == "striped" and source.offchip else []
        common = set.intersection(*(read[entry] for entry in entries)) if len(entries) > 1 else set()
        rest = [item for item in items if item.box not in common]
        hops = self._tree(source, targets, rest, bitwidth, load, adjacency) if rest else {}
        for k, entrance in enumerate(entries if common else ()):
            part = [replace(item, box=_part(item.box, len(entries), k)) for item in items if item.box in common]
            if not box_bits((item.box for item in part), bitwidth):
                continue
            through_entry = adjacency | {source: [(v, link) for v, link in direct if v == entrance]}
            for hop, carried in self._tree(source, targets, part, bitwidth, load, through_entry, spread=True).items():
                hops.setdefault(hop, []).extend(carried)
        return hops

    def _single_entry(  # noqa: PLR0913
        self,
        source: Memory,
        targets: tuple[Memory, ...],
        items: list[DemandItem],
        bitwidth: float,
        load: dict[CommunicationLink, float],
        direct: list[tuple[Memory, CommunicationLink]],
    ) -> dict[Hop, list[DemandItem]]:
        """Of the trees from ``source`` entering over one of the ``direct`` links, the one whose busiest link is busy
        least, then all its links, added to ``load``."""
        options = []
        for v, link in direct:
            trial = dict(load)
            with suppress(nx.NetworkXNoPath):
                entering = self._adjacency() | {source: [(v, link)]}
                hops = self._tree(source, targets, items, bitwidth, trial, entering)
                cycles = [trial[used] / used.bandwidth for used in dict.fromkeys(hop[2] for hop in hops)]
                options.append(((max(cycles, default=0.0), sum(cycles), len(cycles), v), hops, trial))
        if not options:
            raise nx.NetworkXNoPath(f"no route from {source} over one link")
        _, hops, trial = min(options, key=lambda option: option[0])
        load.update(trial)
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
        """Add to ``load`` the links that bring the partial sums ``items`` from their sources to ``target``, the cores
        on the way adding them up, and return the parts of ``items`` each of their hops carries: a Steiner tree over the
        reversed links into the target, every link carrying the sum once. Partial sums in one memory are added by the
        core owning it. No core adds in off-chip memory, so a sum headed there is completed in one of the on-chip
        memories next to it; cut into a part per memory holding partial sums, each completed where it then loads the
        busiest link least and all links least, the reduction becomes a reduce-scatter. No core adds on a bus, so each
        memory sending over it sends its own sum."""
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

        def weight(_: Memory, link: CommunicationLink) -> float:
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
        through it, each link weighted by the cycles the data a target adds keeps it busy, one leaving off-chip
        ``source`` at least as long as its port is then busy sending on all of them. Each step searches once per
        distinct pair of new and carried data among the targets left."""
        bits, cells = _cells(needs, bitwidth)
        held: dict[CommunicationLink, set[tuple]] = {}
        held_bits: dict[CommunicationLink, float] = {}
        sent: set[tuple] = set()
        below: dict[Hop, set[Memory]] = {}
        remaining = set(needs) - {source}
        rate = self.accelerator.offchip_rate(source, READ) if source.offchip else 0.0

        def added(link: CommunicationLink, key: tuple[float, frozenset[tuple]]) -> float:
            fresh, seen = key
            taken = held.get(link, set())
            return fresh + sum(bits[c] for c in seen if c not in taken)

        def search(
            key: tuple[float, frozenset[tuple]],
        ) -> tuple[dict[Memory, float], dict[Memory, tuple[Memory, CommunicationLink]]]:
            def weight(u: Memory, link: CommunicationLink) -> float:
                if not (extra := added(link, key)):
                    return _HOP
                cycles = (load.get(link, 0.0) + held_bits.get(link, 0.0) + extra) / link.bandwidth
                if rate and u is source:
                    port = sum(load.get(out, 0.0) + held_bits.get(out, 0.0) for _, out in adjacency[source])
                    cycles = max(cycles, (port + extra) / rate)
                return cycles + _HOP

            return _dijkstra({source}, adjacency, weight)

        while remaining:
            keys = {t: (sum(bits[c] for c in cells[t] - sent), cells[t] & sent) for t in remaining}
            searches = {key: search(key) for key in set(keys.values())}
            reachable = [t for t in remaining if t in searches[keys[t]][0]]
            if not reachable:
                raise nx.NetworkXNoPath(f"no route from {source} to {sorted(t.name for t in remaining)}")
            target = min(reachable, key=lambda t: (searches[keys[t]][0][t], t))
            node = target
            for u, link in _path(searches[keys[target]][1], target):
                below.setdefault((u, node, link), set()).add(target)
                held_bits[link] = held_bits.get(link, 0.0) + added(link, keys[target])
                held.setdefault(link, set()).update(cells[target])
                node = u
            sent |= cells[target]
            remaining.discard(target)
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
    weight: Callable[[Memory, CommunicationLink], float],
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
            nc = c + weight(u, link)
            if v not in done and nc < cost.get(v, float("inf")):
                cost[v] = nc
                previous[v] = (u, link)
                heapq.heappush(heap, (nc, hops + 1, v))
    return cost, previous
