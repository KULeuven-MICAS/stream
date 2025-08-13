from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice, product
from math import ceil, floor
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
from zigzag.datatypes import Constants, MemoryOperand
from zigzag.hardware.architecture.memory_port import DataDirection

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.utils import intersections
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import SubviewTensor

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.noc.communication_link import CommunicationLink


class CommunicationEvent:
    """
    Represents a communication event between two cores, aggregating one or more CommunicationLinkEvents.
    Tracks sender, receiver, and total energy for the event.
    """

    def __init__(self, id: int, tasks: list["CommunicationLinkEvent"], sender: Core, receiver: Core) -> None:
        # Sanity checks
        assert len(tasks) > 0
        assert all([t.type == tasks[0].type] for t in tasks)
        assert all([t.start == tasks[0].start for t in tasks])
        assert all([t.end == tasks[0].end for t in tasks])
        self.id = id
        self.tasks = tasks
        self.type = tasks[0].type
        self.start = tasks[0].start
        self.end = tasks[0].end
        self.energy = sum([t.energy for t in tasks])
        self.sender = sender
        self.receiver = receiver

    def __str__(self) -> str:
        return (
            f"CommunicationEvent(id={self.id}, sender={self.sender}, receiver={self.receiver}, "
            f"tensor={self.tasks[0].tensors}, energy={self.energy:.2e})"
        )

    def __repr__(self) -> str:
        return str(self)


class CommunicationLinkEvent:
    """Represents an event on a communication link.
    An event has:
        - a type, e.g. "transfer" or "block"
        - a start time
        - an end time
        - a list of tensors relevant for the event:
            * the tensor being transferred
            * the tensor(s) for which we are blocking
        - an activity:
            * the bits per clock cycle used of the link bandwidth
    """

    def __init__(
        self,
        type: str,
        start: int,
        end: int,
        tensors: list[SubviewTensor],
        energy: float,
        activity: float,
        source: Core,
        destinations: list[Core],
    ) -> None:
        self.type = type
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.tensors = tensors
        self.energy = energy
        self.activity = activity
        self.source = source
        self.destinations = destinations

    def __str__(self) -> str:
        return (
            f"CommunicationLinkEvent(type={self.type}, src={self.source}, dests={self.destinations}, "
            f"start={self.start}, end={self.end}, tensors={self.tensors}, "
            f"energy={self.energy:.2e}, activity={self.activity:.2f})"
        )

    def __repr__(self) -> str:
        return str(self)

    def get_operands(self):
        """
        Returns the operand associated with the tensor for this event.
        """
        return [t.layer_operand for t in self.tensors]

    def get_origin(self):
        origins = [tensor.cn_source for tensor in self.tensors]
        assert all([origin == origins[0] for origin in origins])
        return origins[0]


@dataclass(frozen=True, slots=True)
class SharedPrefixPath:
    sources: tuple["Core", ...]
    targets: tuple["Core", ...]
    meeting: "Core"
    paths_from_sources: dict["Core", list["Core"]]
    paths_to_targets: dict["Core", list["Core"]]
    full_paths: dict["Core", list["Core"]]
    total_cost: float
    overlap_edges: int


class MulticastRequest(NamedTuple):
    sources: tuple["Core", ...]
    destinations: tuple["Core", ...]

    def __hash__(self) -> int:
        return hash((self.sources, self.destinations))


@dataclass(frozen=True, slots=True)
class MulticastPathPlan:
    sources: tuple["Core", ...]
    targets: tuple["Core", ...]
    meeting: "Core"
    paths_from_sources: dict["Core", list["Core"]]  # s -> path s..m
    paths_to_targets: dict["Core", list["Core"]]  # t -> path m..t
    full_paths: dict["Core", list["Core"]]  # leaf endpoint -> full path
    total_cost: float
    overlap_edges: int


class CommunicationManager:
    """
    Manages communication events and link usage between cores, including bandwidth normalization and event creation.
    Handles both data transfers and link blocking for memory constraints.
    """

    shortest_paths: dict[tuple[Core, Core], list[Core]]
    events: list[CommunicationEvent]

    def __init__(self, accelerator: "Accelerator") -> None:
        self.accelerator = accelerator
        self.shortest_paths = self.get_shortest_paths()
        self.all_shortest_paths = self.get_all_shortest_paths()
        self.all_pair_links = self.get_all_links_for_all_core_pairs()
        self.events = []
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

    def enumerate_multicast_plans(  # noqa: PLR0912, PLR0915
        self,
        sources: Sequence["Core"],
        targets: Sequence["Core"],
        *,
        weight: str | None = None,
        k_per_leg: int = 2,
        max_meetings: int = 5,
        max_plans: int = 10,
        undirected_signature: bool = False,
    ) -> list[MulticastPathPlan]:
        """
        Return a small, diverse set of high-quality (de-duplicated) path combinations for a single
        multicast/join request. Duplicates are removed when they traverse the same set of edges,
        regardless of meeting node choice.

        Set `undirected_signature=True` if you want (u->v) considered the same as (v->u).
        """
        if not sources:
            raise ValueError("sources must be non-empty")
        if not targets:
            raise ValueError("targets must be non-empty")
        if len(sources) > 1 and len(targets) > 1:
            raise ValueError("only one-to-many or many-to-one is supported")

        G = self.accelerator.cores
        Grev = G.reverse(copy=False) if G.is_directed() else G

        def path_cost(path: list["Core"]) -> float:
            if weight is None:
                return float(len(path) - 1)
            c = 0.0
            for u, v in zip(path, path[1:], strict=False):
                c += float(G.edges[(u, v)].get(weight, 1.0))
            return c

        def _edge_signature(paths: dict["Core", list["Core"]]) -> frozenset:
            if undirected_signature:
                sig: set[frozenset[Core]] = set()
                for p in paths.values():
                    for u, v in zip(p, p[1:], strict=False):
                        sig.add(frozenset((u, v)))
                return frozenset(sig)
            else:
                sig_dir: set[tuple[Core, Core]] = set()
                for p in paths.values():
                    for u, v in zip(p, p[1:], strict=False):
                        sig_dir.add((u, v))
                return frozenset(sig_dir)

        # Distances for meeting-node scoring
        dist_from_sources: dict[Core, dict[Core, float]] = {
            s: nx.single_source_dijkstra_path_length(G, s, weight=weight) for s in sources
        }
        dist_to_targets: dict[Core, dict[Core, float]] = {
            t: nx.single_source_dijkstra_path_length(Grev, t, weight=weight) for t in targets
        }

        # Rank meeting nodes by total Steiner-like cost; tie-break by longer shared segment on the relevant side
        ranked: list[tuple[float, float, Core]] = []
        for m in G.nodes():
            try:
                c_s = sum(dist_from_sources[s][m] for s in sources)
                c_t = sum(dist_to_targets[t][m] for t in targets)
            except KeyError:
                continue
            total = c_s + c_t
            tie = (
                -float(dist_from_sources[sources[0]][m])
                if len(sources) == 1
                else -float(dist_to_targets[targets[0]][m])
            )
            ranked.append((total, tie, m))
        if not ranked:
            raise nx.NetworkXNoPath("no meeting node connects all sources to all targets")

        ranked.sort()
        candidate_ms = [m for _, _, m in islice(ranked, max_meetings)]

        plans: list[MulticastPathPlan] = []
        seen_signatures: set[frozenset] = set()

        for m in candidate_ms:
            if len(sources) == 1:
                s0 = sources[0]
                s_paths = list(islice(nx.shortest_simple_paths(G, s0, m, weight=weight), k_per_leg))
                if not s_paths:
                    continue
                t_paths: dict[Core, list[list[Core]]] = {}
                for t in targets:
                    alt = list(islice(nx.shortest_simple_paths(G, m, t, weight=weight), k_per_leg))
                    if not alt:
                        break
                    t_paths[t] = alt
                if len(t_paths) != len(targets):
                    continue

                for s_path in s_paths:
                    for choice in product(*[t_paths[t] for t in targets]):
                        paths_from_sources = {s0: s_path}
                        paths_to_targets = {t: p for t, p in zip(targets, choice, strict=False)}
                        full_paths = {t: s_path[:-1] + paths_to_targets[t] for t in targets}
                        sig = _edge_signature(full_paths)
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)
                        total = path_cost(s_path) + sum(path_cost(p) for p in choice)
                        overlap = max(0, len(s_path) - 1)
                        plans.append(
                            MulticastPathPlan(
                                sources=(s0,),
                                targets=tuple(targets),
                                meeting=m,
                                paths_from_sources=paths_from_sources,
                                paths_to_targets=paths_to_targets,
                                full_paths=full_paths,
                                total_cost=total,
                                overlap_edges=overlap,
                            )
                        )
            else:
                t0 = targets[0]
                t_paths = list(islice(nx.shortest_simple_paths(G, m, t0, weight=weight), k_per_leg))
                if not t_paths:
                    continue
                s_paths: dict[Core, list[list[Core]]] = {}
                for s in sources:
                    alt = list(islice(nx.shortest_simple_paths(G, s, m, weight=weight), k_per_leg))
                    if not alt:
                        break
                    s_paths[s] = alt
                if len(s_paths) != len(sources):
                    continue

                for t_path in t_paths:
                    for choice in product(*[s_paths[s] for s in sources]):
                        paths_from_sources = {s: p for s, p in zip(sources, choice, strict=False)}
                        paths_to_targets = {t0: t_path}
                        full_paths = {s: paths_from_sources[s][:-1] + t_path for s in sources}
                        sig = _edge_signature(full_paths)
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)
                        total = path_cost(t_path) + sum(path_cost(p) for p in choice)
                        overlap = max(0, len(t_path) - 1)
                        plans.append(
                            MulticastPathPlan(
                                sources=tuple(sources),
                                targets=(t0,),
                                meeting=m,
                                paths_from_sources=paths_from_sources,
                                paths_to_targets=paths_to_targets,
                                full_paths=full_paths,
                                total_cost=total,
                                overlap_edges=overlap,
                            )
                        )

        plans.sort(key=lambda p: (p.total_cost, -p.overlap_edges))
        return plans[:max_plans]

    def get_links_for_multicast_plan(
        self,
        plan: "MulticastPathPlan",
    ) -> tuple["CommunicationLink", ...]:
        """
        Return the unique CommunicationLink objects required to execute a given multicast plan.

        Args:
            plan: A MulticastPathPlan with .full_paths specifying the node sequences.

        Returns:
            A tuple of unique CommunicationLink objects (edge attribute 'cl'),
            sorted for deterministic output.
        """
        G = self.accelerator.cores
        links: set[CommunicationLink] = set()
        for path in plan.full_paths.values():
            for u, v in zip(path, path[1:], strict=False):
                links.add(G.edges[(u, v)]["cl"])
        return tuple(sorted(links))

    def _get_best_shared_prefix_path(  # noqa: PLR0912
        self,
        sources: Sequence["Core"],
        targets: Sequence["Core"],
        weight: str | None = None,
    ) -> SharedPrefixPath:
        """
        Compute routes between multiple sources and multiple targets that are jointly short and
        share as much overlap as possible at the appropriate side:
        - One source, many targets: maximize shared prefix (before branching).
        - Many sources, one target: maximize shared tail (after merging).

        The method finds a meeting node m that minimizes:
            sum_s d(s, m) + sum_t d(m, t)
        and then stitches shortest-path segments through m.

        Args:
            sources: Source cores (non-empty).
            targets: Destination cores (non-empty).
            weight: Optional edge attribute name to use as a weight. If None, treats all edges equally.

        Returns:
            SharedPrefixPath with meeting node, per-source and per-target segments, full paths, cost, and overlap.

        Raises:
            ValueError: If sources or targets is empty, or if both have length > 1 (unsupported pattern).
            networkx.NetworkXNoPath: If no single meeting node connects all sources to all targets.
        """
        if not sources:
            raise ValueError("sources must be a non-empty sequence")
        if not targets:
            raise ValueError("targets must be a non-empty sequence")
        if len(sources) > 1 and len(targets) > 1:
            raise ValueError("This method supports either one-to-many or many-to-one, not many-to-many.")

        G = self.accelerator.cores
        Grev = G.reverse(copy=False) if G.is_directed() else G

        # Distances from each source to all nodes
        dist_from_sources: dict[Core, dict[Core, float]] = {
            s: nx.single_source_dijkstra_path_length(G, s, weight=weight) for s in sources
        }
        # Distances from all nodes to each target (via reverse graph)
        dist_to_targets: dict[Core, dict[Core, float]] = {
            t: nx.single_source_dijkstra_path_length(Grev, t, weight=weight) for t in targets
        }

        # Choose meeting node m
        best_key: tuple[float, float] | None = None
        best_m: Core | None = None

        for m in G.nodes():
            try:
                cost_sources = sum(dist_from_sources[s][m] for s in sources)
                cost_targets = sum(dist_to_targets[t][m] for t in targets)
            except KeyError:
                continue  # unreachable from some source or cannot reach some target
            total = cost_sources + cost_targets

            # Tie-breaker: favor longer shared segment on the appropriate side
            if len(sources) == 1 and len(targets) >= 1:
                tie = -float(dist_from_sources[sources[0]][m])  # deeper prefix
            elif len(targets) == 1 and len(sources) >= 1:
                tie = -float(dist_to_targets[targets[0]][m])  # longer tail
            else:
                tie = 0.0

            key = (total, tie)
            if best_key is None or key < best_key:
                best_key = key
                best_m = m

        if best_m is None:
            raise nx.NetworkXNoPath("No meeting node connects all sources to all targets.")

        # Reconstruct segments
        paths_from_sources: dict[Core, list[Core]] = {s: nx.shortest_path(G, s, best_m, weight=weight) for s in sources}
        paths_to_targets: dict[Core, list[Core]] = {t: nx.shortest_path(G, best_m, t, weight=weight) for t in targets}

        # Build full paths. Keys are the "leaf endpoints":
        # - one-to-many: keys are targets
        # - many-to-one: keys are sources
        full_paths: dict[Core, list[Core]] = {}
        if len(sources) == 1:  # one-to-many
            s0 = sources[0]
            prefix = paths_from_sources[s0]
            for t, m_to_t in paths_to_targets.items():
                full_paths[t] = prefix[:-1] + m_to_t
            overlap_edges = max(0, len(prefix) - 1)
        else:  # many-to-one
            t0 = targets[0]
            tail = paths_to_targets[t0]
            for s, s_to_m in paths_from_sources.items():
                full_paths[s] = s_to_m[:-1] + tail
            overlap_edges = max(0, len(tail) - 1)

        total_cost = sum(dist_from_sources[s][best_m] for s in sources) + sum(
            dist_to_targets[t][best_m] for t in targets
        )

        return SharedPrefixPath(
            sources=tuple(sources),
            targets=tuple(targets),
            meeting=best_m,
            paths_from_sources=paths_from_sources,
            paths_to_targets=paths_to_targets,
            full_paths=full_paths,
            total_cost=total_cost,
            overlap_edges=overlap_edges,
        )

    def compute_multicast_links(
        self,
        sources: Sequence["Core"],
        targets: Sequence["Core"],
        weight: str | None = None,
    ) -> tuple["CommunicationLink", ...]:
        result = self._get_best_shared_prefix_path(sources, targets, weight=weight)
        G = self.accelerator.cores
        required_links: set[CommunicationLink] = set()
        for path in result.full_paths.values():
            for u, v in zip(path, path[1:], strict=False):
                required_links.add(G.edges[(u, v)]["cl"])
        return tuple(sorted(required_links))

    def plan_multicast_sequence(
        self,
        requests: Sequence["MulticastRequest"],
        *,
        initial_mc_cost: float = 1.0,
        reuse_penalty: float = 10.0,
        nb_cols_to_use: int = 2,
        infinite_cost: float = float("inf"),
    ) -> dict["MulticastRequest", tuple["CommunicationLink", ...]]:
        """
        Plan a sequence of multicasts/joins that spreads work by making repeatedly-used links
        progressively more expensive *within this planning run*. Costs are updated only using
        the links selected for the current request.

        Edge routing cost at any step:
            mc_cost = initial_mc_cost + reuse_penalty * uses_run(undirected_link)
        where uses_run counts selections made earlier in this run, treating links as undirected.

        Args:
            requests: Sequence of requests, each with .sources and .destinations.
            initial_mc_cost: Base cost for eligible edges.
            reuse_penalty: Additive penalty per prior use in this run (>= 0).
            nb_cols_to_use: Hard constraint on allowed column indices (inclusive, 1-based).
            infinite_cost: Cost assigned to edges crossing disallowed columns.

        Returns:
            Mapping request -> tuple of CommunicationLink objects used for that request.
        """
        if not requests:
            return {}

        G = self.accelerator.cores

        def undir_key(u: "Core", v: "Core") -> frozenset["Core"]:
            return frozenset((u, v))

        # Per-run usage counts (undirected) used to increase edge costs as we go.
        usage_count: dict[CommunicationLink, int] = {}

        def init_mc_costs() -> None:
            for _, _, d in G.edges(data=True):
                cl = d.get("cl")
                allowed = cl.sender.col_id + 1 <= nb_cols_to_use and cl.receiver.col_id + 1 <= nb_cols_to_use
                d["mc_cost"] = initial_mc_cost if allowed else infinite_cost

        def refresh_mc_costs_from_usage() -> None:
            for _, _, d in G.edges(data=True):
                if d["mc_cost"] == infinite_cost:
                    continue
                cl = d.get("cl")
                uses = usage_count.get(cl, 0)
                d["mc_cost"] = initial_mc_cost + reuse_penalty * uses

        def required_links(paths: dict["Core", list["Core"]]) -> set["CommunicationLink"]:
            links: set[CommunicationLink] = set()
            for path in paths.values():
                for u, v in zip(path, path[1:], strict=False):
                    links.add(G.edges[(u, v)]["cl"])
            return links

        init_mc_costs()
        planned: dict[MulticastRequest, tuple[CommunicationLink, ...]] = {}
        for request in requests:
            # # TEST new function
            # multicast_plans = self.enumerate_multicast_plans(request.sources, request.destinations)
            # for multicast_plan in multicast_plans:
            #     links_used = self.get_links_for_multicast_plan(multicast_plan)
            #     print(multicast_plan)
            #     print(links_used)
            #     print()
            # refresh_mc_costs_from_usage()
            result = self._get_best_shared_prefix_path(request.sources, request.destinations, weight="mc_cost")
            links_used = required_links(result.full_paths)
            # Update usage counts using ONLY links used by THIS request (undirected)
            for link in links_used:
                usage_count[link] = usage_count.get(link, 0) + 1
            planned[request] = tuple(sorted(links_used))
        return planned

    def transfer_tensor(
        self,
        tensor: SubviewTensor,
        sender: Core | int,
        receiver: Core | int,
        receiver_memory_operand: MemoryOperand,
        start_timestep: int,
        duration: int,
        link_bw_fraction: float = 1.0,
    ):
        """
        Transfers a tensor from sender to receiver, possibly using a fraction of the link bandwidth.
        Normalizes bandwidth if total requested exceeds link capacity.
        Creates a CommunicationEvent if the transfer is new across all links.
        """
        assert 0 <= link_bw_fraction <= 1
        end_timestep = start_timestep + duration
        if isinstance(sender, int):
            sender = self.accelerator.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.accelerator.get_core(receiver)
        links = self.get_all_links_for_pair(sender, receiver)
        links = links[0]  # take only the first path
        if not links:  # When sender == receiver
            return 0, 0

        cles = [
            CommunicationLinkEvent(
                type="transfer",
                start=start_timestep,
                end=end_timestep,
                tensors=[tensor],
                energy=duration * link.unit_energy_cost,
                activity=link.bandwidth,
                source=sender,
                destinations=[receiver],
            )
            for link in links
        ]

        link_energy_cost = 0
        is_new_event_across_all_links = True
        for link, cle in zip(links, cles, strict=False):
            transfer_energy_cost, is_new_event = link.transfer(cle)
            if is_new_event:
                link_energy_cost += transfer_energy_cost
            else:
                is_new_event_across_all_links = False
        if is_new_event_across_all_links:
            event = CommunicationEvent(
                id=self.event_id,
                tasks=cles,
                sender=sender,
                receiver=receiver,
            )
            self.events.append(event)
            self.event_id += 1
        # Energy cost of memory reads/writes on sender/receiver
        # For this we need to know the memory operand in order to know where in the sender/receiver the tensor is stored
        # We assume the tensor to be sent is defined from the sender perspective, so we take its operand as the sender
        # memory operand
        sender_memory_operand = tensor.memory_operand
        memory_energy_cost = self.accelerator.get_memory_energy_cost_of_transfer(
            tensor, sender, receiver, sender_memory_operand, receiver_memory_operand
        )
        return link_energy_cost, memory_energy_cost

    def block_offchip_links(
        self,
        too_large_operands: list,
        core_id: int,
        start_timestep: int,
        duration: int,
        node: ComputationNode,
    ):
        """
        Blocks off-chip links for operands that are too large to fit in memory for the duration of the node's execution.
        Handles both output and input operands, and creates CommunicationEvents for new blocks.
        """
        if not too_large_operands:
            return start_timestep
        core = self.accelerator.get_core(core_id)
        assert self.accelerator.offchip_core_id is not None, "Off-chip core id is not set."
        offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
        tensors_per_link: dict[CommunicationLink, list[SubviewTensor]] = {}
        # Determine the flow of data from source to destination depending on the operands
        if Constants.OUTPUT_MEM_OP in too_large_operands:
            source = core
            destinations = [offchip_core]
        else:
            source = offchip_core
            destinations = [core]
        if Constants.OUTPUT_MEM_OP in too_large_operands:
            links_to_offchip = set(self.get_all_links_for_pair(core, offchip_core)[0])  # Take the first path

            for link in links_to_offchip:
                tensors_per_link[link] = tensors_per_link.get(link, []) + [
                    (node.operand_tensors[Constants.OUTPUT_LAYER_OP])
                ]

        # Input operands
        non_output_mem_ops = [op for op in too_large_operands if op != Constants.OUTPUT_MEM_OP]
        if non_output_mem_ops:
            links_from_offchip = set(self.get_all_links_for_pair(offchip_core, core)[0])  # Take the first path
            for link in links_from_offchip:
                tensors_per_link[link] = tensors_per_link.get(link, []) + [
                    node.operand_tensors[node.memory_operand_links.mem_to_layer_op(op)] for op in non_output_mem_ops
                ]

        tensor_bw_per_link = {
            link: [
                (tensor, self.get_instantaneous_offchip_bandwidth(node, tensor.memory_operand))
                for tensor in tensors_this_link
            ]
            for link, tensors_this_link in tensors_per_link.items()
        }

        # Get idle window of the involved links
        block_start = self.get_links_idle_window(tensor_bw_per_link, start_timestep, duration, tensors_per_link)
        # Block them
        for link, req_bw in tensor_bw_per_link.items():
            req_bw_ceiled = ceil(req_bw)
            link.block(
                block_start,
                duration,
                tensors_per_link[link],
                activity=req_bw_ceiled,
                source=source,
                destinations=destinations,
            )
        return block_start

    @staticmethod
    def get_instantaneous_offchip_bandwidth(node: ComputationNode, op: MemoryOperand) -> int:
        """
        Returns the instantaneous off-chip bandwidth for a given operand.
        """
        assert op in node.offchip_bandwidth_per_op
        if op == Constants.OUTPUT_MEM_OP:
            data_dir = DataDirection.WR_IN_BY_LOW
        else:
            data_dir = DataDirection.RD_OUT_TO_LOW
        return node.offchip_bandwidth_per_op[op].get(data_dir)

    def get_links_idle_window(
        self,
        tensor_bw_per_link: dict["CommunicationLink", list[tuple[SubviewTensor, int]]],
        start_timestep: int,
        duration: int,
        tensors_per_link: dict["CommunicationLink", list[SubviewTensor]],
    ) -> int:
        """Return the timestep at which tensor can be transfered across the links.
        Both links must have an idle window large enough for the transfer.
        The timestep must be greater than or equal to best_case_start.

        Args:
            links (dict): CommunicationLinks involved in the transfer and their required bandwidth.
            best_case_start (int): The best case start timestep of the transfer.
            duration (int): The required duration of the idle window.
            tensors (list): The tensors to be transferred. Used to broadcast from previous transfer.
        """
        assert len(tensor_bw_per_link) > 0
        idle_intersections: list[tuple[int, int]] = []
        for i, (link, bandwidth_per_tensor) in enumerate(tensor_bw_per_link.items()):
            # Normalize bandwidth if total requested exceeds link bandwidth
            total_req_bw = sum([bw for _, bw in bandwidth_per_tensor])
            if total_req_bw > link.bandwidth:
                normalization_factor = link.bandwidth / total_req_bw
                bandwidth_per_tensor_floored = [
                    (tensor, floor(normalization_factor * bw)) for tensor, bw in bandwidth_per_tensor
                ]
            else:
                bandwidth_per_tensor_floored = bandwidth_per_tensor
            windows = link.get_idle_window(bandwidth_per_tensor_floored, duration, start_timestep)
            if i == 0:
                idle_intersections = windows
            else:
                idle_intersections = intersections(idle_intersections, windows)
                idle_intersections = [period for period in idle_intersections if period[1] - period[0] >= duration]

        # Note: The earliest window is chosen; if more sophisticated selection is needed, update here.
        earliest_window = idle_intersections[0]
        start_time, _ = earliest_window
        return start_time
