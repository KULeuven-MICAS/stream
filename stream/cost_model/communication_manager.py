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
_MAX_ALLOCATIONS_PER_MEETING = 2
_MAX_MEETINGS = 8
_MAX_POSSIBLE_ALLOCATIONS = 4


def _path_edges(path: list) -> list[tuple]:
    return list(zip(path, path[1:], strict=False))


def _build_edge_weight_map(Gw: nx.Graph, weight_attr: str) -> dict[tuple, float]:
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
    Gw: nx.Graph,
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
    possible_memory_cores: tuple["Core", ...]

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
    meeting: "Core"
    paths_from_sources: dict["Core", list["Core"]]  # s -> path s..m
    paths_to_targets: dict["Core", list["Core"]]  # t -> path m..t
    full_paths: dict["Core", list["Core"]]  # leaf endpoint -> full path
    total_hops_objective: int  # objective used to rank meetings
    overlap_edges: int  # shared prefix (one-to-many) or tail (many-to-one)


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

    def _get_simple_no_meeting_node_plans(
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
            if (is_offchip(u) and is_memory(v)) or (is_offchip(v) and is_memory(u)):
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

            plans.append(
                MulticastPathPlan(
                    sources=tuple(sources),
                    targets=tuple(targets),
                    meeting=None,
                    # For no-meeting, we put everything in full_paths; link extraction will use it.
                    paths_from_sources={},
                    paths_to_targets={},
                    full_paths=st.full_paths,
                    total_hops_objective=st.score,
                    overlap_edges=0,
                )
            )
            if len(plans) >= max_allocations:
                break

        if not plans:
            raise nx.NetworkXNoPath("no feasible no-meeting plan found")

        return plans

    def _get_links_for_multicast_plan(
        self,
        plan: "MulticastPathPlan",
    ) -> tuple["CommunicationLink", ...]:
        """
        Return the unique CommunicationLink objects required to execute a plan.

        Supports:
        - meeting plans: union of edges from paths_from_sources and paths_to_targets
        - no-meeting plans: union of edges from full_paths (keyed by (s,t))
        """
        G = self.accelerator.cores
        links: set[CommunicationLink] = set()

        # No-meeting: use full_paths if present
        if getattr(plan, "meeting", None) is None and getattr(plan, "full_paths", None):
            for path in plan.full_paths.values():
                for u, v in zip(path, path[1:], strict=False):
                    links.add(G.edges[(u, v)]["cl"])
            return tuple(sorted(links))

        # Meeting-based (or fallback placeholder): use both dicts
        for path in plan.paths_from_sources.values():
            for u, v in zip(path, path[1:], strict=False):
                links.add(G.edges[(u, v)]["cl"])

        for path in plan.paths_to_targets.values():
            for u, v in zip(path, path[1:], strict=False):
                links.add(G.edges[(u, v)]["cl"])

        return tuple(sorted(links))

    def _enumerate_multicast_plans(
        self,
        request: "MulticastRequest",
        *,
        offchip_mem_penalty: float = 1000.0,
        k_paths: int = _K_PATHS_PER_TERMINAL,
        beam_width: int = _BEAM_WIDTH,
        max_allocations_per_meeting: int = _MAX_ALLOCATIONS_PER_MEETING,
        max_meetings: int = _MAX_MEETINGS,
    ) -> list["MulticastPathPlan"]:
        """
        Multi-src + multi-dst planner with:
          - meeting constrained to request.possible_memory_cores when provided
          - k-shortest simple paths per terminal (s->m and m->t)
          - beam search to enumerate distinct low-cost unions of edges
        """
        sources = tuple(request.sources)
        targets = tuple(request.destinations)
        if not sources or not targets:
            raise ValueError("sources and targets must be non-empty")

        G = self.accelerator.cores

        # Copy graph and assign weights
        Gw = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)

        offchip_id = self.accelerator.offchip_core_id

        def is_offchip(n: "Core") -> bool:
            return getattr(n, "id", None) == offchip_id

        def is_memory(n: "Core") -> bool:
            return getattr(n, "type", None) == "memory"

        for u, v, d in Gw.edges(data=True):
            if (is_offchip(u) and is_memory(v)) or (is_offchip(v) and is_memory(u)):
                d["w"] = float(offchip_mem_penalty)
            else:
                d["w"] = 1.0

        candidates = tuple(request.possible_memory_cores or ())
        if not candidates:
            return self._get_simple_no_meeting_node_plans(
                sources,
                targets,
                k_paths=_K_PATHS_PER_TERMINAL,
                beam_width=_BEAM_WIDTH,
                max_allocations=_MAX_ALLOCATIONS_PER_MEETING,
                offchip_mem_penalty=offchip_mem_penalty,
            )

        # Precompute edge weights for incremental union scoring (supports directed+undirected)
        edge_w: dict[tuple, float] = _build_edge_weight_map(Gw, "w")

        # For meeting ranking: need dist(s->m) and dist(m->t)
        # dist(m->t) is dist_to_targets[t][m] computed on the reverse graph
        Grev = Gw.reverse(copy=False) if Gw.is_directed() else Gw

        dist_from_sources: dict[Core, dict[Core, float]] = {
            s: nx.single_source_dijkstra_path_length(Gw, s, weight="w") for s in sources
        }
        dist_to_targets: dict[Core, dict[Core, float]] = {
            t: nx.single_source_dijkstra_path_length(Grev, t, weight="w") for t in targets
        }

        INF = float("inf")

        def meeting_objective(m: "Core") -> float:
            return (
                sum(dist_from_sources[s].get(m, INF) for s in sources)
                + sum(dist_to_targets[t].get(m, INF) for t in targets)  # equals dist(m->t) in original
            )

        ranked = sorted(
            ((meeting_objective(m), m) for m in candidates),
            key=lambda x: (x[0], getattr(x[1], "id", 0)),
        )
        top_meetings = [m for score, m in ranked if score < INF][:max_meetings]
        if not top_meetings:
            raise nx.NetworkXNoPath("no meeting node connects all endpoints")

        all_plans: list[MulticastPathPlan] = []

        def push_candidates(
            beam: list[_BeamState],
            key,
            options: list[list["Core"]],
            *,
            into_sources: bool,
            edge_w: dict[tuple, float],
            beam_width: int,
        ) -> list[_BeamState]:
            next_states: list[_BeamState] = []
            for st in beam:
                union_edges = set(st.edges)

                for p in options:
                    ep = _path_edges(p)
                    add_cost = _incremental_union_cost(ep, union_edges, edge_w)
                    new_edges = frozenset(union_edges.union(ep))

                    if into_sources:
                        new_pfs = dict(st.paths_from_sources)
                        new_pfs[key] = p
                        new_pts = st.paths_to_targets  # safe to reuse
                    else:
                        new_pts = dict(st.paths_to_targets)
                        new_pts[key] = p
                        new_pfs = st.paths_from_sources  # safe to reuse

                    next_states.append(
                        _BeamState(
                            edges=new_edges,
                            score=st.score + add_cost,
                            paths_from_sources=new_pfs,
                            paths_to_targets=new_pts,
                        )
                    )

            # Keep best beam_width by score, stabilize with edge-count + repr
            next_states.sort(key=lambda s: (s.score, len(s.edges)))
            return next_states[:beam_width]

        for m in top_meetings:
            # Build k path options for each terminal (meeting must be feasible for all terminals)
            src_options: dict[Core, list[list[Core]]] = {}
            for s in sources:
                # if you cache (s,m) paths you can pass first_path=...
                opts = _k_paths_for_pair_using_cached_first(Gw, s, m, k=k_paths, weight="w", cached_path=None)
                if not opts:
                    src_options = {}
                    break
                src_options[s] = opts
            if not src_options:
                continue

            tgt_options: dict[Core, list[list[Core]]] = {}
            for t in targets:
                opts = _k_paths_for_pair_using_cached_first(Gw, m, t, k=k_paths, weight="w", cached_path=None)
                if not opts:
                    tgt_options = {}
                    break
                tgt_options[t] = opts
            if not tgt_options:
                continue

            # Beam search: first sources, then targets (deterministic)
            beam: list[_BeamState] = [
                _BeamState(edges=frozenset(), score=0.0, paths_from_sources={}, paths_to_targets={})
            ]

            for s in sources:
                beam = push_candidates(
                    beam,
                    s,
                    src_options[s],
                    into_sources=True,
                    edge_w=edge_w,
                    beam_width=beam_width,
                )
                if not beam:
                    break
            if not beam:
                continue

            for t in targets:
                beam = push_candidates(
                    beam,
                    t,
                    tgt_options[t],
                    into_sources=False,
                    edge_w=edge_w,
                    beam_width=beam_width,
                )
                if not beam:
                    break
            if not beam:
                continue

            # Emit up to max_allocations_per_meeting distinct unions
            seen_edge_sets: set[frozenset[tuple]] = set()
            for st in beam:
                # Defensive: ensure full coverage
                if len(st.paths_from_sources) != len(sources):
                    continue
                if len(st.paths_to_targets) != len(targets):
                    continue

                if st.edges in seen_edge_sets:
                    continue
                seen_edge_sets.add(st.edges)

                all_plans.append(
                    MulticastPathPlan(
                        sources=sources,
                        targets=targets,
                        meeting=m,
                        paths_from_sources=st.paths_from_sources,
                        paths_to_targets=st.paths_to_targets,
                        full_paths={},  # keep if your dataclass requires it
                        total_hops_objective=st.score,
                        overlap_edges=0,
                    )
                )
                if len(seen_edge_sets) >= max_allocations_per_meeting:
                    break

        if not all_plans:
            raise nx.NetworkXNoPath("no feasible meeting plan found")

        # Global sort so results are stable (and good plans first)
        all_plans.sort(
            key=lambda p: (
                float(getattr(p, "total_hops_objective", 0.0)),
                getattr(getattr(p, "meeting", None), "id", 0),
            )
        )
        return all_plans

    def get_possible_resource_allocations(
        self,
        src_allocs: list["Core"],
        dst_allocs: list["Core"],
        possible_memory_cores: tuple["Core", ...],
    ) -> tuple[tuple["CommunicationLink", ...], ...]:
        request = MulticastRequest(
            sources=src_allocs,  # type: ignore
            destinations=dst_allocs,  # type: ignore
            possible_memory_cores=possible_memory_cores,
        )

        multicast_plans = self._enumerate_multicast_plans(request)

        possible_paths: set[tuple[CommunicationLink, ...]] = set()
        for multicast_plan in multicast_plans:
            links_used = self._get_links_for_multicast_plan(multicast_plan)
            possible_paths.add(links_used)

        # Stable output ordering
        return tuple(sorted(possible_paths, key=lambda x: (len(x), repr(x)))[:_MAX_POSSIBLE_ALLOCATIONS])
