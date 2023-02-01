from math import ceil
from typing import List
import networkx as nx
from networkx import DiGraph
import itertools
from stream.classes.hardware.architecture.communication_link import CommunicationLink

from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from stream.classes.workload.tensor import Tensor


class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    In this Stream version, the cores are actually a graph with directed edges representing communication links.
    """

    def __init__(self, name, cores: DiGraph, global_buffer: MemoryInstance or None, offchip_core_id=None):
        self.name = name
        self.cores = cores
        self.global_buffer = global_buffer
        if offchip_core_id is None:
            self.offchip_core_id = max((core.id for core in self.cores.nodes()))
        else:
            self.offchip_core_id = offchip_core_id
        self.shortest_paths = self.get_shortest_paths()
        self.pair_links = self.get_links_for_all_core_pairs()

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name,
                "cores": self.cores}

    def get_core(self, core_id: int or str) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        core = next((core for core in self.cores.nodes() if core.id == core_id), None)
        if not core:
            raise ValueError(f"Requested core with id {core_id} is not present in accelerator.")
        return core

    def get_shortest_paths(self):
        # For each core pair save a shortest path
        shortest_paths = {}
        for (producer_core, consumer_core) in itertools.product(self.cores.nodes(), self.cores.nodes()):
            shortest_paths[(producer_core, consumer_core)] = nx.shortest_path(self.cores, producer_core, consumer_core)
        return shortest_paths

    def get_links_for_all_core_pairs(self):
        communication_links = {}
        for pair, path in self.shortest_paths.items():
            traversed_edges = [(i, j) for i, j in zip(path, path[1:])]
            communication_links[pair] = [self.cores.edges[traversed_edge]['cl'] for traversed_edge in traversed_edges]
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
        sender = self.get_core(sender_id)
        receiver = self.get_core(receiver_id)
        return self.get_links_for_pair(sender, receiver)

    def transfer_data(self, tensor: Tensor, sender: Core or int, receiver: Core or int, receiver_memory_operand: str, start_timestep: int) -> tuple[int, int, float, float]:
        """Transfer a data tensor from sender to receiver for this accelerator starting at timestep.

        Args:
            tensor (Tensor): The tensor to be transferred.
            sender (Core): The sending core.
            receiver (Core): The receiving core.
            receiver_memory_operand (str): The memory operand storing the tensor on the receiving end of the transfer.
            start_timestep (int): The timestep at which to start the data transfer.


        Returns:
            int: The timestep at which the transfer is complete.
        """
        link_energy_cost = 0
        if isinstance(sender, int):
            sender = self.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.get_core(receiver)
        links: List[CommunicationLink] = self.get_links_for_pair(sender, receiver)
        if not links:  # if links is empty (happens when sender == receiver, i.e. "transfer" from Core A -> Core A)
            return start_timestep, start_timestep, link_energy_cost, 0  # the "transfer" doesn't require any time
        transfer_start = max(start_timestep, links[0].available_from)
        transfer_timestep = transfer_start
        for link in links:
            transfer_timestep, transfer_energy_cost = link.put(tensor, transfer_timestep)
            link_energy_cost += transfer_energy_cost
        transfer_end = transfer_timestep  # Timestep of last transfer complete
        # Energy cost of memory reads/writes on sender/receiver
        # For this we need to know the memory operand in order to know where in the sender/receiver the tensor is stored
        # We assume the tensor to be sent is defined from the sender perspective, so we take its operand as the sender memory operand
        sender_memory_operand = tensor.memory_operand
        memory_energy_cost = self.get_memory_energy_cost_of_transfer(tensor, sender, receiver, sender_memory_operand, receiver_memory_operand)
        return transfer_start, transfer_end, link_energy_cost, memory_energy_cost

    def get_memory_energy_cost_of_transfer(self, tensor: Tensor, sender: Core or int, receiver: Core or int, sender_memory_operand: str, receiver_memory_operand: str):

        # Convert given sender and receiver to Core object if given as ids
        if isinstance(sender, int):
            sender = self.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.get_core(receiver)

        # Get the top level of output memory for the sender and the top level of input memory for the consumer_operand
        sender_top_memory_level = sender.memory_hierarchy.get_operand_top_level(sender_memory_operand)
        receiver_top_memory_level = receiver.memory_hierarchy.get_operand_top_level(receiver_memory_operand)
        # Sender memory energy
        nb_sender_memory_reads_for_data = ceil(tensor.size / sender_top_memory_level.read_bw)
        sender_energy = sender_top_memory_level.read_energy * nb_sender_memory_reads_for_data
        # Receiver memory energy
        nb_receiver_memory_writes_for_data = ceil(tensor.size / sender_top_memory_level.write_bw)
        receiver_energy = receiver_top_memory_level.write_energy * nb_receiver_memory_writes_for_data

        return sender_energy + receiver_energy

    def block_offchip_links(self, too_large_operands, core_id, start_timestep, duration, cn_id) -> int:
        """Block the communication link between 'core' and the offchip core starting at timestep 'start_timestep' for duration 'duration'.

        Args:
            too_large_operands (list): List of insufficient memory operands. This decides which link to block
            core_id (int): The core id.
            start_timestep (int): The ideal start timestep of the blocking.
            duration (int): The duration of the blocking in cycles.
            cn_id (tuple): The computational node id.
        """
        links_to_block = []
        effective_start_timestep = start_timestep
        core = self.get_core(core_id)
        offchip_core = self.get_core(self.offchip_core_id)
        if 'O' in too_large_operands:
            links_to_block.append(self.get_links_for_pair(core, offchip_core))
        if [op for op in too_large_operands if op != 'O']:
            links_to_block.append(self.get_links_for_pair(offchip_core, core))
        if not too_large_operands:
            return start_timestep
        # Get the worst case start time of all the links for all the operands
        worst_case_start_time = max([link.available_from for links in links_to_block for link in links])
        worst_case_start_time = max(start_timestep, worst_case_start_time)
        links_set = set((link for links in links_to_block for link in links))
        for link in links_set:
            blocking_start_timestep, blocking_end_timestep = link.block(worst_case_start_time, duration, cn_id)
            assert blocking_start_timestep == worst_case_start_time, "Mismatch between worst case link start time and effective link block start time."
        # for links in links_to_block:
        #     for link in links: # There can be multiple links if the offchip is not directly connected to this core
        #         blocking_start_timestep, blocking_end_timestep = link.block(worst_case_start_time, duration, cn_id)
        #         assert blocking_start_timestep == worst_case_start_time, "Mismatch between worst case link start time and effective link block start time."
        return worst_case_start_time
