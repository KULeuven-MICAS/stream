import itertools
from collections.abc import Sequence
from dataclasses import dataclass
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
    source: "Core"
    targets: tuple["Core", ...]
    meeting: "Core"
    prefix: list["Core"]
    branches: dict["Core", list["Core"]]
    full_paths: dict["Core", list["Core"]]
    total_cost: float
    overlap_edges: int


class MulticastRequest(NamedTuple):
    source: "Core"
    destinations: tuple["Core", ...]


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
        for producer_core, consumer_core in itertools.product(self.accelerator.core_list, self.accelerator.core_list):
            shortest_paths[(producer_core, consumer_core)] = self.accelerator.cores.shortest_path(
                producer_core, consumer_core
            )
        return shortest_paths

    def get_all_shortest_paths(self) -> dict[tuple[Core, Core], list[list[Core]]]:
        """Return a dictionary with all shortest paths between all core pairs."""
        all_shortest_paths: dict[tuple[Core, Core], list[list[Core]]] = {}
        for producer_core, consumer_core in itertools.product(self.accelerator.core_list, self.accelerator.core_list):
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

    def _get_best_shared_prefix_path(
        self,
        source: "Core",
        targets: Sequence["Core"],
        weight: str | None = None,
    ) -> SharedPrefixPath:
        """
        Compute routes from one source to multiple targets that are jointly short
        and share as much prefix as possible.

        The method finds a meeting node m that minimizes:
            d(source, m) + sum_i d(m, targets[i])
        The shared prefix for all routes is the path source -> m.

        Args:
            source: Start core.
            targets: Destination cores (non-empty).
            weight: Optional edge attribute name to use as a weight. If None, treats all edges equally.

        Returns:
            SharedPrefixResult with the chosen meeting node, shared prefix, per-target branches,
            full paths per target, total cost, and overlap size.

        Raises:
            ValueError: If targets is empty.
            networkx.NetworkXNoPath: If no node connects source to all targets.
        """
        if not targets:
            raise ValueError("targets must be a non-empty sequence")

        G = self.accelerator.cores

        dist_from_source = nx.single_source_dijkstra_path_length(G, source, weight=weight)

        if G.is_directed():
            Grev = G.reverse(copy=False)
        else:
            Grev = G

        dist_to_targets: dict[Core, dict[Core, float]] = {
            t: nx.single_source_dijkstra_path_length(Grev, t, weight=weight) for t in targets
        }

        best_key: tuple[float, float] | None = None  # (total_cost, -dist_from_source[m]) for tie-breaking
        best_m: Core | None = None

        for m in G.nodes():
            if m not in dist_from_source:
                continue
            try:
                total_cost = dist_from_source[m] + sum(dist_to_targets[t][m] for t in targets)
            except KeyError:
                continue
            key = (total_cost, -float(dist_from_source[m]))
            if best_key is None or key < best_key:
                best_key = key
                best_m = m

        if best_m is None:
            raise nx.NetworkXNoPath("No meeting node connects source to all targets.")

        prefix = nx.shortest_path(G, source, best_m, weight=weight)

        branches: dict[Core, list[Core]] = {}
        full_paths: dict[Core, list[Core]] = {}
        for t in targets:
            branch = nx.shortest_path(G, best_m, t, weight=weight)
            branches[t] = branch
            full_paths[t] = prefix[:-1] + branch

        total_cost = dist_from_source[best_m] + sum(dist_to_targets[t][best_m] for t in targets)
        overlap_edges = max(0, len(prefix) - 1)

        return SharedPrefixPath(
            source=source,
            targets=tuple(targets),
            meeting=best_m,
            prefix=prefix,
            branches=branches,
            full_paths=full_paths,
            total_cost=total_cost,
            overlap_edges=overlap_edges,
        )

    def compute_multicast_links(
        self,
        source: "Core",
        targets: Sequence["Core"],
        weight: str | None = None,
    ) -> tuple["CommunicationLink", ...]:
        best_shared_prefix_path = self._get_best_shared_prefix_path(source, targets, weight=weight)
        G = self.accelerator.cores
        required_links: set[CommunicationLink] = set()
        for path in best_shared_prefix_path.full_paths.values():
            for u, v in zip(path, path[1:], strict=False):
                required_links.add(G.edges[(u, v)]["cl"])
        return tuple(sorted(required_links))

    def plan_multicast_sequence(
        self,
        requests: Sequence[MulticastRequest],
        *,
        initial_mc_cost: float = 1.0,
        reuse_penalty: float = 1.0,
        nb_cols_to_use: int = 2,
        infinite_cost: float = float("inf"),
    ) -> dict[MulticastRequest, tuple["CommunicationLink", ...]]:
        """
        Plan a sequence of multicast transfers to minimize the number of *new* CommunicationLinks
        activated across all transfers, encouraging reuse of links selected earlier.

        The planner proceeds greedily. At each step, among the remaining requests, it selects the one
        that introduces the fewest new links given the current active set (ties broken by lower total cost).
        For each selection, edge costs are reweighted to prefer already active links via `reuse_factor`
        and to penalize new links via `new_link_penalty`.

        Args:
            requests: Sequence of (source, targets) groups to multicast.
            initial_mc_cost: Initial weight for multicast cost of each edge.
            reuse_penalty: Multiplicative factor (>= 1.0 recommended) applied to the base cost for used links.

        Returns:
            A list of (result, required_links) per request in the planned execution order.
        """
        if not requests:
            return {}

        G = self.accelerator.cores
        active_links: set[CommunicationLink] = set()
        remaining: list[MulticastRequest] = list(requests)
        planned: list[set[CommunicationLink]] = []
        requests_order: list[MulticastRequest] = []

        def init_mc_costs() -> None:
            for _, _, d in G.edges(data=True):
                cl = d.get("cl")
                if cl.sender.col_id + 1 > nb_cols_to_use or cl.receiver.col_id + 1 > nb_cols_to_use:
                    d["mc_cost"] = infinite_cost
                else:
                    d["mc_cost"] = initial_mc_cost

        def update_mc_costs(links_used) -> None:
            for _, _, d in G.edges(data=True):
                link_used = d.get("cl") in links_used
                if not link_used:
                    continue
                d["mc_cost"] += reuse_penalty

        def compute_required_links(paths: dict["Core", list["Core"]]) -> set["CommunicationLink"]:
            links: set[CommunicationLink] = set()
            for path in paths.values():
                for u, v in zip(path, path[1:], strict=False):
                    links.add(G.edges[(u, v)]["cl"])
            return links

        init_mc_costs()
        while remaining:
            req = remaining.pop(0)
            result = self._get_best_shared_prefix_path(req.source, req.destinations, weight="mc_cost")
            required_links = compute_required_links(result.full_paths)
            update_mc_costs(required_links)
            active_links.update(required_links)
            planned.append(required_links)
            requests_order.append(req)

        planned_sorted: dict[MulticastRequest, tuple[CommunicationLink, ...]] = {}
        # Sort the different required_links in planned
        for req, links in zip(requests_order, planned, strict=False):
            planned_sorted[req] = tuple(sorted(links))

        return planned_sorted

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
