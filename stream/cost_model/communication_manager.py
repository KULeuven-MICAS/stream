from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator


# Tune these defaults as needed
_K_PATHS_PER_TERMINAL = 4  # k in k-shortest
_BEAM_WIDTH = 8  # B in beam search
_MAX_ALLOCATIONS_PER_MEETING = 1
_MAX_MEETINGS = 8
_MAX_POSSIBLE_ALLOCATIONS = 6


def _path_edges(path: list) -> list[tuple]:
    return list(zip(path, path[1:], strict=False))


def _build_edge_weight_map(Gw: nx.Graph, weight_attr: str) -> dict[tuple, float]:  # noqa: N803
    """
    Map directed edge (u,v) -> weight.
    For undirected graphs, also add (v,u) so scoring works regardless of direction.
    """
    wmap: dict[tuple, float] = {}
    for u, v, d in Gw.edges(data=True):
        w = float(d.get(weight_attr, 1.0))
        wmap[(u, v)] = w
        if not Gw.is_directed():
            wmap[(v, u)] = w
    return wmap


def _incremental_union_cost(edge_list: list[tuple], union_edges: set[tuple], edge_w: dict[tuple, float]) -> float:
    add = 0.0
    for e in edge_list:
        if e not in union_edges:
            add += edge_w[e]
    return add


def _k_paths_for_pair_using_cached_first(
    Gw: nx.Graph,  # noqa: N803
    s,
    t,
    *,
    k: int,
    weight: str,
    cached_path: list | None,
) -> list[list]:
    """
    Build up to k simple shortest paths for (s,t), preferring cached_path if provided.
    Uses networkx.shortest_simple_paths to get alternatives (Yen-like).
    """
    out: list[list] = []
    seen: set[tuple] = set()

    if cached_path:
        p = list(cached_path)
        out.append(p)
        seen.add(tuple(p))
        if len(out) >= k:
            return out

    if s == t:
        if (s,) not in seen:
            out.append([s])
        return out[:k]

    # shortest_simple_paths might not accept weight on older nx
    try:
        gen = nx.shortest_simple_paths(Gw, s, t, weight=weight)
    except TypeError:
        gen = nx.shortest_simple_paths(Gw, s, t)

    for p in gen:
        tp = tuple(p)
        if tp in seen:
            continue
        out.append(list(p))
        seen.add(tp)
        if len(out) >= k:
            break
    return out


@dataclass(frozen=True)
class _BeamStateNoMeeting:
    edges: frozenset[tuple]  # union of edges for all chosen pairs so far
    score: float  # sum of weights of unique edges
    full_paths: dict  # (s,t) -> path


@dataclass(frozen=True)
class _BeamState:
    # Union of directed edges (u,v) in the allocation so far
    edges: frozenset[tuple]
    # Accumulated cost (sum of weights of unique edges)
    score: float
    # Chosen paths
    paths_from_sources: dict
    paths_to_targets: dict


class MulticastRequest(NamedTuple):
    sources: tuple["Core", ...]
    destinations: tuple["Core", ...]

    def __hash__(self) -> int:
        return hash((self.sources, self.destinations))


@dataclass(frozen=True, slots=True)
class UnicastPathPlan:
    source: "Core"
    target: "Core"
    full_paths: list["Core"]  # Core sequence


@dataclass(frozen=True, slots=True)
class MulticastPathPlan:
    sources: tuple["Core", ...]
    targets: tuple["Core", ...]
    total_hops_objective: int  # objective used to rank meetings
    links_used: tuple["CommunicationLink", ...]


class CommunicationManager:
    """
    Manages communication events and link usage between cores, including bandwidth normalization and event creation.
    Handles both data transfers and link blocking for memory constraints.
    """

    shortest_paths: dict[tuple[Core, Core], list[Core]]

    def __init__(self, accelerator: "Accelerator") -> None:
        self.accelerator = accelerator
        self.shortest_paths = self.get_shortest_paths()
        self.all_shortest_paths = self.get_all_shortest_paths()
        self.all_pair_links = self.get_all_links_for_all_core_pairs()
        self.event_id = 0

    def get_shortest_paths(self):
        # For each core pair save a shortest path
        shortest_paths: dict[tuple[Core, Core], list[Core]] = {}
        for producer_core, consumer_core in product(self.accelerator.core_list, self.accelerator.core_list):
            shortest_paths[(producer_core, consumer_core)] = self.accelerator.cores.shortest_path(
                producer_core, consumer_core
            )
        return shortest_paths

    def get_all_shortest_paths(self) -> dict[tuple[Core, Core], list[list[Core]]]:
        """Return a dictionary with all shortest paths between all core pairs."""
        all_shortest_paths: dict[tuple[Core, Core], list[list[Core]]] = {}
        for producer_core, consumer_core in product(self.accelerator.core_list, self.accelerator.core_list):
            paths = list(nx.all_shortest_paths(self.accelerator.cores, producer_core, consumer_core))
            all_shortest_paths[(producer_core, consumer_core)] = paths
        return all_shortest_paths

    def get_all_links_for_all_core_pairs(self):
        communication_links: dict[tuple[Core, Core], tuple[tuple[CommunicationLink], ...]] = {}
        for pair, paths in self.all_shortest_paths.items():
            links: list[tuple[CommunicationLink]] = []
            for path in paths:
                traversed_edges = [(i, j) for i, j in zip(path, path[1:], strict=False)]
                links.append(
                    tuple(self.accelerator.cores.edges[traversed_edge]["cl"] for traversed_edge in traversed_edges)
                )
            communication_links[pair] = tuple(links)
        return communication_links

    def get_all_links_for_pair(self, sender: Core, receiver: Core) -> tuple[tuple["CommunicationLink"], ...]:
        """Return the list of traversed CommunicationLinks for sending data from sender core to receiver core.

        Args:
            sender_id (Core): the sending core
            receiver_id (Core): the receiving core
        """
        return self.all_pair_links[(sender, receiver)]

    def get_all_links(self):
        """Return all unique CommunicationLinks."""
        return list(set(d["cl"] for _, _, d in self.accelerator.cores.edges(data=True)))

    def _get_simple_no_meeting_node_plans(  # noqa: PLR0912, PLR0915
        self,
        sources: tuple["Core", ...],
        targets: tuple["Core", ...],
        *,
        k_paths: int = 3,
        beam_width: int = 32,
        max_allocations: int = 32,
        offchip_mem_penalty: float = 1000.0,
    ) -> list["MulticastPathPlan"]:
        """
        No-meeting enumeration that covers ALL sources and ALL destinations together.

        We construct a plan by choosing one path per (source,target) pair (default: all pairs),
        then taking the union of edges (CommunicationLinks) used by those paths.

        Enumeration:
        - up to k_paths per pair via shortest_simple_paths (with cached shortest_paths as first)
        - beam search over unions to avoid cartesian explosion
        """

        if not sources or not targets:
            raise ValueError("sources and targets must be non-empty")

        G = self.accelerator.cores
        Gw = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)

        offchip_id = self.accelerator.offchip_core_id

        def is_offchip(n: "Core") -> bool:
            return getattr(n, "id", None) == offchip_id

        def is_memory(n: "Core") -> bool:
            return getattr(n, "type", None) == "memory"

        # Weighting (same idea as meeting case, but applied to no-meeting too)
        for u, v, d in Gw.edges(data=True):
            if is_offchip(v) and is_memory(u):
                d["w"] = float(offchip_mem_penalty)
            else:
                d["w"] = 1.0

        edge_w = _build_edge_weight_map(Gw, "w")

        # Define the required connectivity.
        # Default: all-pairs coverage (each source can reach each destination).
        pairs: list[tuple[Core, Core]] = [(s, t) for s in sources for t in targets]

        # Build path options for each pair.
        pair_options: dict[tuple[Core, Core], list[list[Core]]] = {}
        for s, t in pairs:
            cached = self.shortest_paths.get((s, t))
            opts = _k_paths_for_pair_using_cached_first(Gw, s, t, k=k_paths, weight="w", cached_path=cached)
            if not opts:
                # If any required pair is disconnected, no plan exists under this definition.
                raise nx.NetworkXNoPath(f"no path for required pair {s}->{t}")
            pair_options[(s, t)] = opts

        # Beam search over pairs (no cartesian product).
        beam: list[_BeamStateNoMeeting] = [_BeamStateNoMeeting(edges=frozenset(), score=0.0, full_paths={})]

        # Deterministic order helps reproducibility
        def _pair_sort_key(st_pair: tuple["Core", "Core"]) -> tuple:
            s, t = st_pair
            return (getattr(s, "id", 0), getattr(t, "id", 0))

        for s, t in sorted(pairs, key=_pair_sort_key):
            options = pair_options[(s, t)]
            next_states: list[_BeamStateNoMeeting] = []

            for st in beam:
                base_edges = set(st.edges)
                for p in options:
                    pe = _path_edges(p)
                    add_cost = _incremental_union_cost(pe, base_edges, edge_w)
                    new_edges = frozenset(base_edges.union(pe))
                    new_full = dict(st.full_paths)
                    new_full[(s, t)] = p
                    next_states.append(
                        _BeamStateNoMeeting(edges=new_edges, score=st.score + add_cost, full_paths=new_full)
                    )

            next_states.sort(key=lambda st: (st.score, len(st.edges)))
            beam = next_states[:beam_width]
            if not beam:
                raise nx.NetworkXNoPath("beam search eliminated all states; try larger beam_width or k_paths")

        # Emit unique edge-sets as MulticastPathPlans
        plans: list[MulticastPathPlan] = []
        seen_edge_sets: set[frozenset[tuple]] = set()

        for st in beam:
            if len(st.full_paths) != len(pairs):
                continue
            if st.edges in seen_edge_sets:
                continue
            seen_edge_sets.add(st.edges)

            # Add the links used for this plan
            links_used: list[CommunicationLink] = list()
            for path in st.full_paths.values():
                for u, v in zip(path, path[1:], strict=False):
                    cl = G.edges[(u, v)]["cl"]
                    if cl not in links_used:
                        links_used.append(cl)

            plans.append(
                MulticastPathPlan(
                    sources=tuple(sources),
                    targets=tuple(targets),
                    total_hops_objective=st.score,
                    links_used=tuple(links_used),
                )
            )
            if len(plans) >= max_allocations:
                break

        if not plans:
            raise nx.NetworkXNoPath("no feasible no-meeting plan found")

        return plans

    def _enumerate_multicast_plans(
        self,
        request: "MulticastRequest",
        *,
        offchip_mem_penalty: float = 1000.0,
        k_paths: int = _K_PATHS_PER_TERMINAL,
        beam_width: int = _BEAM_WIDTH,
        max_allocations: int = _MAX_ALLOCATIONS_PER_MEETING,
    ) -> list["MulticastPathPlan"]:
        """
        Simple multi-src + multi-dst planner without meeting points.
        Returns plans with direct paths from each source to each destination.
        """
        sources = tuple(request.sources)
        targets = tuple(request.destinations)

        return self._get_simple_no_meeting_node_plans(
            sources,
            targets,
            k_paths=k_paths,
            beam_width=beam_width,
            max_allocations=max_allocations,
            offchip_mem_penalty=offchip_mem_penalty,
        )

    def get_possible_transfer_plan(
        self,
        src_allocs: list["Core"],
        dst_allocs: list["Core"],
    ) -> tuple[MulticastPathPlan, ...]:
        request = MulticastRequest(
            sources=src_allocs,  # type: ignore
            destinations=dst_allocs,  # type: ignore
        )
        multicast_plans = self._enumerate_multicast_plans(request, offchip_mem_penalty=1)
        # Stable output ordering
        return tuple(multicast_plans)
