import itertools
from math import ceil
from typing import TYPE_CHECKING

from zigzag.datatypes import Constants, MemoryOperand

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.utils import intersections
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import Tensor

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.noc.communication_link import CommunicationLink


class CommunicationEvent:
    """Represents a communication event involving one or more CommunicationLinks."""

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
        return f"CommunicationEvent(id={self.id}, sender={self.sender}, receiver={self.receiver})"

    def __repr__(self) -> str:
        return str(self)


class CommunicationLinkEvent:
    """Represents an event on a communication link.
    An event has:
        - a type, e.g. "transfer" or "block"
        - a start time
        - an end time
        - a tensors relevant for the event:
            * the tensor being transferred
            * the tensor for which we are blocking
        - an activity:
            * the bits per clock cycle used of the link bandwidth
    """

    def __init__(
        self,
        type: str,
        start: int,
        end: int,
        tensor: Tensor,
        energy: float,
        activity: float,
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
        return self.tensor.layer_operand

    def get_origin(self):
        return self.tensor.origin


class CommunicationManager:
    """Manages the inter-core and offchip communication of an Accelerator."""

    shortest_paths: dict[tuple[Core, Core], list[Core]]
    events: list[CommunicationEvent]

    def __init__(self, accelerator: "Accelerator") -> None:
        self.accelerator = accelerator
        self.shortest_paths = self.get_shortest_paths()
        self.pair_links = self.get_links_for_all_core_pairs()
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

    def get_links_for_all_core_pairs(self):
        communication_links: dict[tuple[Core, Core], "CommunicationLink"] = {}
        for pair, path in self.shortest_paths.items():
            traversed_edges = [(i, j) for i, j in zip(path, path[1:])]
            communication_links[pair] = [
                self.accelerator.cores.edges[traversed_edge]["cl"] for traversed_edge in traversed_edges
            ]
        return communication_links

    def get_links_for_pair(self, sender: Core, receiver: Core) -> list["CommunicationLink"]:
        """Return the list of traversed CommunicationLinks for sending data from sender core to receiver core.

        Args:
            sender_id (Core): the sending core
            receiver_id (Core): the receiving core
        """
        return self.pair_links[(sender, receiver)]

    def get_all_links(self):
        """Return all unique CommunicationLinks."""
        return list(set(d["cl"] for _, _, d in self.accelerator.cores.edges(data=True)))

    def update_links(
        self,
        tensor: Tensor,
        sender: Core | int,
        receiver: Core | int,
        receiver_memory_operand: MemoryOperand,
        start_timestep: int,
        duration: int,
        link_bw_fraction: float = 1.0,
    ) -> tuple[float, float]:
        """Update the links for transfer of a tensor between sender and receiver core at a given timestep.
        A CommunicationEvent is created containing one or more CommunicationLinkEvents,
        i.e. one CommunicationLinkEvent per involved CommunicationLink.

        Args:
            tensor (Tensor): The tensor to be transferred.
            sender (Core): The sending core.
            receiver (Core): The receiving core.
            receiver_memory_operand (str): The memory operand storing the tensor on the receiving end of the transfer.
            start_timestep: The timestep at which to start the data transfer.
            duration: Duration of the transfer

        Returns:
            tuple: A tuple containing the link and memory energy costs associated with this transfer.
        """
        end_timestep = start_timestep + duration
        if isinstance(sender, int):
            sender = self.accelerator.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.accelerator.get_core(receiver)
        links = self.get_links_for_pair(sender, receiver)
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
        event = CommunicationEvent(
            id=self.event_id,
            tasks=cles,
            sender=sender,
            receiver=receiver,
        )
        self.events.append(event)
        self.event_id += 1

        link_energy_cost = 0
        for link, cle in zip(links, cles):
            transfer_energy_cost = link.transfer(cle)
            link_energy_cost += transfer_energy_cost
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
        too_large_operands: list[MemoryOperand],
        core_id: int,
        start_timestep: int,
        duration: int,
        node: ComputationNode,
    ) -> int:
        """Block the communication link between 'core' and the offchip core starting at timestep 'start_timestep' for
        duration 'duration'.

        Args:
            too_large_operands: List of insufficient memory operands. This decides which links to block
            core_id: The core id.
            start_timestep: The ideal start timestep of the blocking.
            duration: The duration of the blocking in cycles.
            node: The computational node for which we are blocking the links.
        """

        def get_inst_bw(op: MemoryOperand) -> int:
            assert op in node.offchip_bandwidth_per_op
            if op == Constants.OUTPUT_MEM_OP:
                return node.offchip_bandwidth_per_op[op].wr_in_by_low
            return node.offchip_bandwidth_per_op[op].rd_out_to_low

        if not too_large_operands:
            return start_timestep
        core = self.accelerator.get_core(core_id)
        assert self.accelerator.offchip_core_id is not None, "Off-chip core id is not set."
        offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
        tensors_per_link: dict["CommunicationLink", list[Tensor]] = {}

        # Output operand
        if Constants.OUTPUT_MEM_OP in too_large_operands:
            links_to_offchip = set(self.get_links_for_pair(core, offchip_core))

            for link in links_to_offchip:
                tensors_per_link[link] = [(node.operand_tensors[Constants.OUTPUT_LAYER_OP])]

        # Input operands
        non_output_mem_ops = [op for op in too_large_operands if op != Constants.OUTPUT_MEM_OP]
        if non_output_mem_ops:
            links_from_offchip = set(self.get_links_for_pair(offchip_core, core))
            for link in links_from_offchip:
                tensors_per_link[link] = [
                    node.operand_tensors[node.memory_operand_links.mem_to_layer_op(op)] for op in non_output_mem_ops
                ]

        # Sum the required bandwidth for all tensors on each link
        total_required_bw_per_link = {
            link: sum([get_inst_bw(tensor.memory_operand) for tensor in tensors])
            for link, tensors in tensors_per_link.items()
        }

        # Get idle window of the involved links
        block_start = self.get_links_idle_window(total_required_bw_per_link, start_timestep, duration, tensors_per_link)

        # # Block them
        for link, tensors in tensors_per_link.items():
            operands = [tensor.memory_operand for tensor in tensors]
            bandwidths = [get_inst_bw(op) for op in operands]
            senders = [core if operand == Constants.OUTPUT_MEM_OP else offchip_core for operand in operands]
            receivers = [offchip_core if operand == Constants.OUTPUT_MEM_OP else core for operand in operands]
            link.block(
                block_start, duration, tensors, bandwidth_per_tensor=bandwidths, senders=senders, receivers=receivers
            )

        return block_start

    def get_links_idle_window(
        self,
        links: dict["CommunicationLink", int],
        best_case_start: int,
        duration: int,
        tensors_per_link: dict["CommunicationLink", list[Tensor]],
    ) -> int:
        """Return the timestep at which tensor can be transfered across the links.
        Both links must have an idle window large enough for the transfer.
        The timestep must be greater than or equal to best_case_start.

        Args:
            links (dict): CommunicationLinks involved in the transfer and their required bandwidth.
            best_case_start: The best case start timestep of the transfer.
            duration: The required duration of the idle window.
            tensors: The tensors to be transferred. Used to broadcast from previous transfer.
        """
        assert len(links) > 0
        idle_intersections: list[tuple[int, int]] = []
        for i, (link, req_bw) in enumerate(links.items()):
            req_bw = min(req_bw, link.bandwidth)  # ceil the bw
            windows = link.get_idle_window(req_bw, duration, best_case_start, tensors_per_link[link])
            if i == 0:
                idle_intersections = windows
            else:
                idle_intersections = intersections(idle_intersections, windows)
                idle_intersections = [period for period in idle_intersections if period[1] - period[0] >= duration]
        return idle_intersections[0][0]
