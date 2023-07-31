import itertools
import networkx as nx

from zigzag.classes.hardware.architecture.core import Core
from stream.classes.workload.tensor import Tensor
from stream.classes.hardware.architecture.utils import intersections


class CommunicationEvent:
    """Represents a communication event involving one or more CommunicationLinks."""

    def __init__(self, id, tasks) -> None:
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

    def __str__(self) -> str:
        return f"CommunicationEvent(id={self.id})"

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
    """

    def __init__(self, type, start, end, tensors, energy) -> None:
        self.type = type
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.tensors = tensors
        self.energy = energy

    def __str__(self) -> str:
        return f"CommunicationLinkEvent(type={self.type}, start={self.start}, end={self.end}, tensors={self.tensors}, energy={self.energy:.2e})"

    def __repr__(self) -> str:
        return str(self)

    def get_operands(self):
        return [tensor.layer_operand for tensor in self.tensors]

    def get_origin(self):
        origins = [tensor.origin for tensor in self.tensors]
        assert all([origin == origins[0] for origin in origins])
        return origins[0]


class CommunicationManager:
    """Manages the inter-core and offchip communication of an Accelerator."""

    def __init__(self, accelerator) -> None:
        self.accelerator = accelerator
        self.shortest_paths = self.get_shortest_paths()
        self.pair_links = self.get_links_for_all_core_pairs()
        self.events = []
        self.event_id = 0

    def get_shortest_paths(self):
        # For each core pair save a shortest path
        shortest_paths = {}
        for producer_core, consumer_core in itertools.product(
            self.accelerator.cores.nodes(), self.accelerator.cores.nodes()
        ):
            shortest_paths[(producer_core, consumer_core)] = nx.shortest_path(
                self.accelerator.cores, producer_core, consumer_core
            )
        return shortest_paths

    def get_links_for_all_core_pairs(self):
        communication_links = {}
        for pair, path in self.shortest_paths.items():
            traversed_edges = [(i, j) for i, j in zip(path, path[1:])]
            communication_links[pair] = [
                self.accelerator.cores.edges[traversed_edge]["cl"]
                for traversed_edge in traversed_edges
            ]
            # print(pair, communication_links[pair])
        return communication_links

    def get_links_for_pair(self, sender, receiver):
        """Return the list of traversed CommunicationLinks for sending data from sender core to receiver core.

        Args:
            sender_id (Core): the sending core
            receiver_id (Core): the receiving core
        """
        return self.pair_links[(sender, receiver)]

    def get_links_for_pair_id(self, sender_id, receiver_id):
        """Return the list of traversed CommunicationLinks for sending data from sender core to receiver core.

        Args:
            sender_id (int): the sending core id
            receiver_id (int): the receiving core id
        """
        # Find the sender and receiver based on the given ids
        sender = self.shortest_paths = self.get_shortest_paths().get_core(sender_id)
        receiver = self.accelerator.get_core(receiver_id)
        return self.get_links_for_pair(sender, receiver)

    def get_all_links(self):
        """Return all unique CommunicationLinks."""
        return list(set(d["cl"] for _, _, d in self.accelerator.cores.edges(data=True)))

    def update_links(
        self,
        tensor: Tensor,
        sender: Core or int,
        receiver: Core or int,
        receiver_memory_operand: str,
        start_timestep: int,
        duration: int,
    ) -> tuple[int, int, float, float]:
        """Update the links for transfer of a tensor between sender and receiver core at a given timestep.
        A CommunicationEvent is created containing one or more CommunicationLinkEvents,
        i.e. one CommunicationLinkEvent per involved CommunicationLink.

        Args:
            tensor (Tensor): The tensor to be transferred.
            sender (Core): The sending core.
            receiver (Core): The receiving core.
            receiver_memory_operand (str): The memory operand storing the tensor on the receiving end of the transfer.
            start_timestep (int): The timestep at which to start the data transfer.
            duration (int): Duration of the transfer

        Returns:
            int: The timestep at which the transfer is complete.
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
                tensors=[tensor],
                energy=duration * link.unit_energy_cost,
            )
            for link in links
        ]
        event = CommunicationEvent(
            id=self.event_id,
            tasks=cles,
        )
        self.events.append(event)
        self.event_id += 1

        link_energy_cost = 0
        for link, cle in zip(links, cles):
            transfer_energy_cost = link.transfer(cle)
            link_energy_cost += transfer_energy_cost
        # Energy cost of memory reads/writes on sender/receiver
        # For this we need to know the memory operand in order to know where in the sender/receiver the tensor is stored
        # We assume the tensor to be sent is defined from the sender perspective, so we take its operand as the sender memory operand
        sender_memory_operand = tensor.memory_operand
        memory_energy_cost = self.accelerator.get_memory_energy_cost_of_transfer(
            tensor, sender, receiver, sender_memory_operand, receiver_memory_operand
        )
        return link_energy_cost, memory_energy_cost

    def block_offchip_links(
        self, too_large_operands, core_id, start_timestep, duration, cn
    ) -> int:
        """Block the communication link between 'core' and the offchip core starting at timestep 'start_timestep' for duration 'duration'.

        Args:
            too_large_operands (list): List of insufficient memory operands. This decides which links to block
            core_id (int): The core id.
            start_timestep (int): The ideal start timestep of the blocking.
            duration (int): The duration of the blocking in cycles.
            cn (ComputationNode): The computational node for which we are blocking the links.
        """
        links_to_block = set()
        core = self.accelerator.get_core(core_id)
        offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
        if "O" in too_large_operands:
            links_to_block.update(set(self.get_links_for_pair(core, offchip_core)))
        if [op for op in too_large_operands if op != "O"]:
            links_to_block.update(set(self.get_links_for_pair(offchip_core, core)))
        if not too_large_operands:
            return start_timestep
        links_to_block = list(links_to_block)
        # Get idle window of the involved links
        block_start = self.get_links_idle_window(
            links_to_block, start_timestep, duration
        )
        # Get the tensors for which we are blocking based on the operands
        tensors = []
        for mem_op in too_large_operands:
            layer_op = next(
                k for k, v in cn.memory_operand_links.items() if v == mem_op
            )
            tensors.append(cn.operand_tensors[layer_op])
        for link in links_to_block:
            link.block(block_start, duration, tensors)
        return block_start

    def get_links_idle_window(
        self, links: list, best_case_start: int, duration: int
    ) -> int:
        """Return the timestep at which tensor can be transfered across the links.
        Both links must have an idle window large enough for the transfer.
        The timestep must be greater than or equal to best_case_start.

        Args:
            links (list): Set of the CommunicationLinks involved in the transfer.
            best_case_start (int): The best case start timestep of the transfer.
            duration (int): The required duration of the idle window.
        """
        assert len(links) > 0

        link = links[0]
        idle_intersections = link.idle_periods
        idle_intersections = [
            period for period in idle_intersections if period[1] - period[0] >= duration
        ]

        for link in links[1:]:
            idle_intersections = intersections(idle_intersections, link.idle_periods)
            idle_intersections = [
                period
                for period in idle_intersections
                if period[1] - period[0] >= duration
            ]
        # Pick the first idle intersection that satisfied all the constraints
        if not idle_intersections:
            raise ValueError(f"There is no overlapping idle time for {links}.")
        # Get the first idle window that has a long enough duration
        # when taking into account the best case start timestep.
        for idle_window in idle_intersections:
            start = max(idle_window[0], best_case_start)
            end = idle_window[1]
            if end - start >= duration:
                return start
        raise ValueError(
            "There is no long enough idle period with a late enough start time."
        )
