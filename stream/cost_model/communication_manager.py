from dataclasses import dataclass
from itertools import product
from math import ceil, floor
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
from zigzag.datatypes import Constants, MemoryOperand
from zigzag.hardware.architecture.memory_port import DataDirection

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.utils import intersections
from stream.workload.tensor import SubviewTensor


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

    def get_unicast_plan_no_memory_core(self, source: "Core", target: "Core") -> UnicastPathPlan:
        assert nx.has_path(self.accelerator.cores, source, target), f"No path between {source} and {target}"
        path = self.shortest_paths[(source, target)]
        return UnicastPathPlan(source=source, target=target, full_paths=path)

    def _get_simple_no_meeting_node_plans(
        self, sources: tuple["Core", ...], targets: tuple["Core", ...]
    ) -> list[MulticastPathPlan]:
        # Fallback to simple paths stored in the communication manager
        plans = []
        for source in sources:
            for target in targets:
                path = self.shortest_paths[(source, target)]
                if not path:
                    continue
                full_paths = {target: path}
                plans.append(
                    MulticastPathPlan(
                        sources=(source,),
                        targets=(target,),
                        meeting=None,
                        paths_from_sources={source: path},
                        paths_to_targets={target: path},
                        full_paths=full_paths,
                        total_hops_objective=len(path) - 1,
                        overlap_edges=0,
                    )
                )
        return plans

    def enumerate_multicast_plans(
        self,
        request: MulticastRequest,
        *,
        offchip_mem_penalty: float = 1000.0,
    ) -> list[MulticastPathPlan]:
        """
        Minimal planner with weighted distances to avoid offchip hops:
        - Meeting points are restricted to memory cores, excluding the offchip core, and limited to columns in use.
        - Distances/paths use weights:
            * weight = 1 for normal edges
            * weight = `offchip_mem_penalty` for edges between the offchip core and any memory core
        This strongly discourages paths that bounce via offchip.
        - Broadcast (one source, many targets): rank meetings by sum of weighted distances m->targets.
        - Join (many sources, one target):     rank meetings by sum of weighted distances sources->m.
        - For each selected meeting, build weighted-shortest paths and return one plan per meeting.

        The original graph is NOT modified.
        """
        sources = request.sources
        targets = request.destinations
        if not sources or not targets:
            raise ValueError("sources and targets must be non-empty")
        if len(sources) > 1 and len(targets) > 1:
            raise ValueError("only one-to-many or many-to-one is supported")

        G = self.accelerator.cores

        # Work on a COPY so we don't touch original edge attributes
        Gw = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)

        offchip_id = self.accelerator.offchip_core_id

        def is_offchip(n: "Core") -> bool:
            return getattr(n, "id", None) == offchip_id

        def is_memory(n: "Core") -> bool:
            return getattr(n, "type", None) == "memory"

        # Assign weights on the copy
        for u, v, d in Gw.edges(data=True):
            if (is_offchip(u) and is_memory(v)) or (is_offchip(v) and is_memory(u)):
                d["w"] = float(offchip_mem_penalty)
            else:
                d["w"] = 1.0

        Grev = Gw.reverse(copy=False) if Gw.is_directed() else Gw  # for distances *to* targets

        # Candidate meeting nodes: memory tiles, not offchip, and in used columns
        candidates = request.possible_memory_cores
        if not candidates:
            plans = self._get_simple_no_meeting_node_plans(sources, targets)
            return plans

        # Precompute weighted distances
        dist_from_sources: dict[Core, dict[Core, float]] = {
            s: nx.single_source_dijkstra_path_length(Gw, s, weight="w") for s in sources
        }
        dist_to_targets: dict[Core, dict[Core, float]] = {
            t: nx.single_source_dijkstra_path_length(Grev, t, weight="w") for t in targets
        }

        INF = float("inf")

        def objective(m: "Core") -> float:
            if len(sources) == 1:
                # broadcast: minimize sum weighted distances meeting->targets
                return sum(dist_to_targets[t].get(m, INF) for t in targets)
            else:
                # join: minimize sum weighted distances sources->meeting
                return sum(dist_from_sources[s].get(m, INF) for s in sources)

        # Rank meetings and keep the best few (stable tie-break on node id if present)
        ranked = sorted(((objective(m), m) for m in candidates), key=lambda x: (x[0], getattr(x[1], "id", 0)))
        top_meetings = [m for score, m in ranked if score < INF]
        if not top_meetings:
            raise nx.NetworkXNoPath("no meeting node connects all endpoints")

        plans: list[MulticastPathPlan] = []
        for m in top_meetings:
            # Build weighted shortest paths on the COPY
            paths_from_sources = {s: nx.shortest_path(Gw, s, m, weight="w") for s in sources}
            paths_to_targets = {t: nx.shortest_path(Gw, m, t, weight="w") for t in targets}

            # Stitch full paths keyed by leaf endpoints
            full_paths: dict[Core, list[Core]] = {}
            if len(sources) == 1:
                s0 = sources[0]
                prefix = paths_from_sources[s0]
                for t, m_to_t in paths_to_targets.items():
                    full_paths[t] = prefix[:-1] + m_to_t
                overlap_edges = max(0, len(prefix) - 1)
            else:
                t0 = targets[0]
                tail = paths_to_targets[t0]
                for s, s_to_m in paths_from_sources.items():
                    full_paths[s] = s_to_m[:-1] + tail
                overlap_edges = max(0, len(tail) - 1)

            plans.append(
                MulticastPathPlan(
                    sources=tuple(sources),
                    targets=tuple(targets),
                    meeting=m,
                    paths_from_sources=paths_from_sources,
                    paths_to_targets=paths_to_targets,
                    full_paths=full_paths,
                    total_hops_objective=int(objective(m)),  # weighted objective now
                    overlap_edges=overlap_edges,
                )
            )

        return plans

    def get_links_for_unicast_plan(
        self,
        plan: "UnicastPathPlan",
    ) -> tuple["CommunicationLink", ...]:
        """
        Return the unique CommunicationLink objects required to execute a given unicast plan.

        Args:
            plan: A UnicastPlan with .full_paths specifying the node sequences.

        Returns:
            A tuple of unique CommunicationLink objects (edge attribute 'cl'),
            sorted for deterministic output.
        """
        G = self.accelerator.cores
        links: set[CommunicationLink] = set()
        for u, v in zip(plan.full_paths, plan.full_paths[1:], strict=False):
            links.add(G.edges[(u, v)]["cl"])
        return tuple(sorted(links))

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
