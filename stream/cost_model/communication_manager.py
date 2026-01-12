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
_BEAM_WIDTH = 32  # B in beam search
_MAX_ALLOCATIONS_PER_MEETING = 4
_MAX_MEETINGS = 8
_MAX_POSSIBLE_ALLOCATIONS = 16


def _iter_k_shortest_simple_paths(
    G: nx.Graph,
    src,
    dst,
    *,
    k: int,
    weight: str | None,
) -> list[list]:
    """
    Return up to k loopless paths from src to dst ordered by total weight.
    Uses networkx.shortest_simple_paths (Yen-like). If weight is unsupported
    by the installed NetworkX version, falls back to unweighted.
    """
    if src == dst:
        return [[src]]

    try:
        gen = nx.shortest_simple_paths(G, src, dst, weight=weight)
    except TypeError:
        # Older NetworkX versions did not accept "weight=" here
        gen = nx.shortest_simple_paths(G, src, dst)

    paths: list[list] = []
    for p in gen:
        paths.append(list(p))
        if len(paths) >= k:
            break
    return paths


def _path_edge_pairs(path: list) -> list[tuple]:
    return list(zip(path, path[1:], strict=False))


def _edge_weight_sum_new(
    edge_pairs: list[tuple],
    *,
    current_edges: set[tuple],
    edge_w: dict[tuple, float],
) -> float:
    """
    Incremental score: add weights only for edges not already present.
    """
    add = 0.0
    for e in edge_pairs:
        if e not in current_edges:
            add += edge_w[e]
    return add


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
        self, sources: tuple["Core", ...], targets: tuple["Core", ...]
    ) -> list["MulticastPathPlan"]:
        # Fallback: per-(s,t) single shortest path as before
        plans = []
        for source in sources:
            for target in targets:
                path = self.shortest_paths.get((source, target))
                if not path:
                    continue
                plans.append(
                    MulticastPathPlan(
                        sources=(source,),
                        targets=(target,),
                        meeting=None,
                        paths_from_sources={source: path},
                        paths_to_targets={target: path},  # placeholder so link extraction works
                        full_paths={},  # keep field if your dataclass expects it
                        total_hops_objective=len(path) - 1,
                        overlap_edges=0,
                    )
                )
        return plans

    def _get_links_for_multicast_plan(
        self,
        plan: "MulticastPathPlan",
    ) -> tuple["CommunicationLink", ...]:
        """
        Union links from both sides:
          sources -> meeting (or direct)
          meeting -> targets (or direct placeholder)
        """
        G = self.accelerator.cores
        links: set[CommunicationLink] = set()

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
            return self._get_simple_no_meeting_node_plans(sources, targets)

        # Precompute edge weights for incremental union scoring.
        # Important: store the directed edge (u,v) as key exactly as produced by paths.
        edge_w: dict[tuple, float] = {}
        for u, v, d in Gw.edges(data=True):
            edge_w[(u, v)] = float(d.get("w", 1.0))

        # For meeting ranking: need dist(s->m) and dist(m->t)
        # dist(m->t) is dist_to_target[t][m] using reverse graph single-source from t
        Grev = Gw.reverse(copy=False) if Gw.is_directed() else Gw

        dist_from_sources: dict[Core, dict[Core, float]] = {
            s: nx.single_source_dijkstra_path_length(Gw, s, weight="w") for s in sources
        }
        dist_to_targets: dict[Core, dict[Core, float]] = {
            t: nx.single_source_dijkstra_path_length(Grev, t, weight="w") for t in targets
        }

        INF = float("inf")

        def meeting_objective(m: "Core") -> float:
            a = 0.0
            for s in sources:
                a += dist_from_sources[s].get(m, INF)
            for t in targets:
                a += dist_to_targets[t].get(m, INF)  # equals dist(m->t) in original
            return a

        ranked = sorted(
            ((meeting_objective(m), m) for m in candidates),
            key=lambda x: (x[0], getattr(x[1], "id", 0)),
        )
        top_meetings = [m for score, m in ranked if score < INF][:max_meetings]
        if not top_meetings:
            raise nx.NetworkXNoPath("no meeting node connects all endpoints")

        all_plans: list[MulticastPathPlan] = []

        for m in top_meetings:
            # Build k path options for each terminal
            # If any terminal has 0 paths, this meeting is infeasible.
            src_options: dict[Core, list[list[Core]]] = {}
            for s in sources:
                opts = _iter_k_shortest_simple_paths(Gw, s, m, k=k_paths, weight="w")
                if not opts:
                    src_options = {}
                    break
                src_options[s] = opts
            if not src_options:
                continue

            tgt_options: dict[Core, list[list[Core]]] = {}
            for t in targets:
                opts = _iter_k_shortest_simple_paths(Gw, m, t, k=k_paths, weight="w")
                if not opts:
                    tgt_options = {}
                    break
                tgt_options[t] = opts
            if not tgt_options:
                continue

            # Beam search over terminals: first sources, then targets (deterministic order)
            init = _BeamState(
                edges=frozenset(),
                score=0.0,
                paths_from_sources={},
                paths_to_targets={},
            )
            beam: list[_BeamState] = [init]

            def push_candidates(
                beam: list[_BeamState],
                key,
                options: list[list["Core"]],
                into_sources: bool,
            ) -> list[_BeamState]:
                next_states: list[_BeamState] = []
                for st in beam:
                    current_edges = set(st.edges)
                    for p in options:
                        ep = _path_edge_pairs(p)
                        add_cost = _edge_weight_sum_new(ep, current_edges=current_edges, edge_w=edge_w)
                        new_edges = frozenset(current_edges.union(ep))

                        if into_sources:
                            new_pfs = dict(st.paths_from_sources)
                            new_pfs[key] = p
                            new_pts = st.paths_to_targets
                        else:
                            new_pts = dict(st.paths_to_targets)
                            new_pts[key] = p
                            new_pfs = st.paths_from_sources

                        next_states.append(
                            _BeamState(
                                edges=new_edges,
                                score=st.score + add_cost,
                                paths_from_sources=new_pfs,
                                paths_to_targets=new_pts,
                            )
                        )

                # Keep best beam_width by score, tie-break by size and stable repr
                next_states.sort(
                    key=lambda s: (
                        s.score,
                        len(s.edges),
                    )
                )
                return next_states[:beam_width]

            for s in sources:
                beam = push_candidates(beam, s, src_options[s], into_sources=True)
                if not beam:
                    break

            if not beam:
                continue

            for t in targets:
                beam = push_candidates(beam, t, tgt_options[t], into_sources=False)
                if not beam:
                    break

            if not beam:
                continue

            # Emit up to max_allocations_per_meeting plans for this meeting
            # Ensure uniqueness by edge set.
            seen_edge_sets: set[frozenset[tuple]] = set()
            for st in beam:
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
                        full_paths={},  # optional; keep if your dataclass still has it
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
