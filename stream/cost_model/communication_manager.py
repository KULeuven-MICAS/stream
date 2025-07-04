import itertools
from math import ceil, floor
from typing import TYPE_CHECKING

import networkx as nx
from zigzag.datatypes import Constants, MemoryOperand
from zigzag.hardware.architecture.memory_port import DataDirection

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.utils import intersections
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import Tensor

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
            f"tensor={self.tasks[0].tensor}, energy={self.energy:.2e})"
        )

    def __repr__(self) -> str:
        return str(self)


class CommunicationLinkEvent:
    """
    Represents a communication event on a single link, including sender, receiver, tensor, and activity.
    """

    def __init__(
        self,
        type: str,
        start: int,
        end: int,
        tensor: Tensor,
        energy: float,
        activity: int,
        sender: Core,
        receiver: Core,
    ) -> None:
        self.type = type
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.tensor = tensor
        self.energy = energy
        self.activity = activity
        self.sender = sender
        self.receiver = receiver

    def __str__(self) -> str:
        return (
            f"CommunicationLinkEvent(type={self.type}, start={self.start}, end={self.end}, tensor={self.tensor}, "
            f"energy={self.energy:.2e}, activity={self.activity:.2f}, sender={self.sender}, receiver={self.receiver})"
        )

    def __repr__(self) -> str:
        return str(self)

    def get_operands(self):
        """
        Returns the operand associated with the tensor for this event.
        """
        return self.tensor.layer_operand

    def get_origin(self):
        """
        Returns the origin node of the tensor for this event.
        """
        return self.tensor.origin


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

    def transfer_tensor(
        self,
        sender: Core,
        receiver: Core,
        tensor: Tensor,
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
                tensor=tensor,
                energy=duration * link.unit_energy_cost * link_bw_fraction,
                activity=ceil(link_bw_fraction * link.bandwidth),
                sender=sender,
                receiver=receiver,
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
        tensors_per_link: dict[CommunicationLink, list[Tensor]] = {}

        # Output operand
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
        block_start = self.get_links_idle_window(tensor_bw_per_link, start_timestep, duration)

        # # Block them
        for link, tensor_bws in tensor_bw_per_link.items():
            for tensor, bandwidth in tensor_bws:
                operand = tensor.memory_operand
                sender = core if operand == Constants.OUTPUT_MEM_OP else offchip_core
                receiver = offchip_core if operand == Constants.OUTPUT_MEM_OP else core
                cle, is_new_event = link.block(
                    block_start, duration, tensor, bandwidth=bandwidth, sender=sender, receiver=receiver
                )
                if is_new_event:
                    event = CommunicationEvent(
                        id=self.event_id,
                        tasks=[cle],
                        sender=sender,
                        receiver=receiver,
                    )
                    self.events.append(event)
                    self.event_id += 1
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
        tensor_bw_per_link: dict["CommunicationLink", list[tuple[Tensor, int]]],
        start_timestep: int,
        duration: int,
    ):
        """
        Finds the earliest idle window for all involved links, normalizing bandwidth if needed.
        Returns the start time for the transfer/block.
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
