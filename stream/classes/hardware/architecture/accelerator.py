from math import ceil
from typing import List
import networkx as nx
from networkx import DiGraph
import itertools
from stream.classes.cost_model.memory_manager import MemoryManager
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

    def __init__(
        self,
        name,
        cores: DiGraph,
        offchip_core_id=None,
    ):
        self.name = name
        self.cores = cores
        self.offchip_core_id = offchip_core_id
        self.shortest_paths = self.get_shortest_paths()
        self.pair_links = self.get_links_for_all_core_pairs()
        self.memory_manager = MemoryManager(self)

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.cores}

    def get_core(self, core_id: int or str) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        core = next((core for core in self.cores.nodes() if core.id == core_id), None)
        if not core:
            raise ValueError(
                f"Requested core with id {core_id} is not present in accelerator."
            )
        return core

    def get_shortest_paths(self):
        # For each core pair save a shortest path
        shortest_paths = {}
        for producer_core, consumer_core in itertools.product(
            self.cores.nodes(), self.cores.nodes()
        ):
            shortest_paths[(producer_core, consumer_core)] = nx.shortest_path(
                self.cores, producer_core, consumer_core
            )
        return shortest_paths

    def get_links_for_all_core_pairs(self):
        communication_links = {}
        for pair, path in self.shortest_paths.items():
            traversed_edges = [(i, j) for i, j in zip(path, path[1:])]
            communication_links[pair] = [
                self.cores.edges[traversed_edge]["cl"]
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
        sender = self.get_core(sender_id)
        receiver = self.get_core(receiver_id)
        return self.get_links_for_pair(sender, receiver)

    def transfer_data(
        self,
        tensor: Tensor,
        sender: Core or int,
        receiver: Core or int,
        receiver_memory_operand: str,
        start_timestep: int,
    ) -> tuple[int, int, float, float]:
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
        if (
            not links
        ):  # if links is empty (happens when sender == receiver, i.e. "transfer" from Core A -> Core A)
            return (
                start_timestep,
                start_timestep,
                link_energy_cost,
                0,
            )  # the "transfer" doesn't require any time
        transfer_start = max(start_timestep, links[0].available_from)
        for link in links:
            transfer_end, transfer_energy_cost = link.put(tensor, transfer_start)
            link_energy_cost += transfer_energy_cost
        # Energy cost of memory reads/writes on sender/receiver
        # For this we need to know the memory operand in order to know where in the sender/receiver the tensor is stored
        # We assume the tensor to be sent is defined from the sender perspective, so we take its operand as the sender memory operand
        sender_memory_operand = tensor.memory_operand
        memory_energy_cost = self.get_memory_energy_cost_of_transfer(
            tensor, sender, receiver, sender_memory_operand, receiver_memory_operand
        )
        return transfer_start, transfer_end, link_energy_cost, memory_energy_cost

    def transfer_tensor_to_core(
        self,
        tensor: Tensor,
        receiving_core_id: int,
        tensor_operand: str,
        non_evictable_tensors: list,
        worst_case_timestep: int,
    ):
        """Transfer a tensor to a given core id.
        This function computes when the transfer can take place based on four factors:
        1) The timestep from which this tensor is available for transfer on a sender core.
        2) When the communication link in charge of these transfers are ready.
        3) When the receiving core has enough space to store the tensor.
        4) The transfer is scheduled as close to the computation as possible. This prevents the
        whole memory from being filled up with data that is only required in the far future.

        Args:
            tensor (Tensor): The tensor to transfer.
            receiving_core_id (int): The id of the core that needs to receive the tensor.
            tensor_operand (str): The memory operand where the tensor needs to be stored.
            non_evictable_tensors (list): the stored tensor that cannot be evicted
            worst_case_timestep (int): when the data cannot be prefetched (no enough space), the latest timestep that it needs to be transferred
        """
        ## STEP 0: Check if the tensor is already on the receiving core
        # Get the top instance where the tensor will be transferred to
        receiving_core = self.get_core(receiving_core_id)
        receiving_core_instances = set(
            self.memory_manager.top_instances[receiving_core]
        )
        receiving_top_instance = self.get_top_instance_of_core(
            receiving_core_id, tensor_operand
        )
        if self.memory_manager.contains(tensor, receiving_top_instance):
            return -1, 0, 0, 0, 0, False
        ## STEP 1: Since when is the tensor available on a sending core
        # Get the top instances storing the tensor
        (
            instances_storing_tensor,
            stored_since_timesteps,
        ) = self.find_tensor_in_top_instances(tensor)
        # Pick the core that has stored the tensor the longest
        stored_since_timestep = min(stored_since_timesteps.values())
        storing_instance = next(
            (
                top_instance
                for (top_instance, timestep) in stored_since_timesteps.items()
                if timestep == stored_since_timestep
            )
        )

        ## STEP 2: Since when are the links available for the transfer
        sender_cores = self.memory_manager.cores_per_top_instance[storing_instance]
        # TODO If the storing_instance is a shared instance across more than one core,
        # TODO there will be multiple possible cores to transfer between.
        # TODO For now, we take the first one
        sender_core = sender_cores[0]
        links = self.get_links_for_pair(sender_core, receiving_core)
        # TODO: Currently, we just select the first shortest-distance communication link.
        link_available_timestep = max([link.available_from for link in links])
        data_transfer_duration = max(
            [ceil(tensor.size / link.bandwidth) for link in links]
        )

        consider_transfer_from_timestep = max(
            stored_since_timestep, link_available_timestep
        )
        worst_case_timestep = consider_transfer_from_timestep
        ## STEP 3: When the receiving core has enough space to store the tensor (don't consider the data eviction)
        can_transfer_from_timestep = self.memory_manager.test_add_tensor_to_core(
            tensor,
            receiving_core_id,
            consider_transfer_from_timestep,
            worst_case_timestep,
            data_transfer_duration,
            memory_op=tensor_operand,
        )
        can_end_from_timestep = can_transfer_from_timestep + data_transfer_duration

        ## STEP 4: Add it to the correct memory (consider the data eviction, thus get the actual available transfer time: evictions_complete_timestep)
        (
            evictions_complete_timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = self.memory_manager.add_tensor_to_core(
            tensor,
            receiving_core_id,
            can_transfer_from_timestep,
            can_end_from_timestep,
            non_evictable_tensors,
            memory_op=tensor_operand,
        )
        actual_available_transfer_start = evictions_complete_timestep

        ## STEP 5: Transfer the data
        (
            transfer_start,
            transfer_end,
            transfer_link_energy_cost,
            transfer_memory_energy_cost,
        ) = self.transfer_data(
            tensor,
            sender_core.id,
            receiving_core_id,
            tensor_operand,
            actual_available_transfer_start,
        )

        ## STEP 6: Check if the already transfered data can be removed from the sender core (excluding DRAM)
        # As long as the sender core no longer need it, we wipe it up to save space for other data fetching and prefetching.
        if sender_core.id == self.offchip_core_id:
            pass
        else:
            if (storing_instance not in tensor.instance_priorities) or (
                tensor.instance_priorities[storing_instance] == 0
            ):
                top_level_idx = self.memory_manager.get_top_level_idx(
                    sender_core, tensor.memory_operand
                )
                self.memory_manager.remove_tensor_from_top_instance(
                    storing_instance,
                    tensor,
                    transfer_end,
                    write_back_to_offchip=False,
                )

        ## STEP 7: Give back a flag that tells us whether this transfer came from off-chip for energy tracking
        came_from_offchip = sender_core.id == self.offchip_core_id

        return (
            transfer_end,
            transfer_link_energy_cost,
            transfer_memory_energy_cost,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
            came_from_offchip,
        )

    def get_memory_energy_cost_of_transfer(
        self,
        tensor: Tensor,
        sender: Core or int,
        receiver: Core or int,
        sender_memory_operand: str,
        receiver_memory_operand: str,
    ):
        # Convert given sender and receiver to Core object if given as ids
        if isinstance(sender, int):
            sender = self.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.get_core(receiver)

        # Get the top level of output memory for the sender and the top level of input memory for the consumer_operand
        sender_top_memory_level = sender.memory_hierarchy.get_operand_top_level(
            sender_memory_operand
        )
        receiver_top_memory_level = receiver.memory_hierarchy.get_operand_top_level(
            receiver_memory_operand
        )
        # Sender memory energy
        nb_sender_memory_reads_for_data = ceil(
            tensor.size / sender_top_memory_level.read_bw
        )
        sender_energy = (
            sender_top_memory_level.read_energy * nb_sender_memory_reads_for_data
        )
        # Receiver memory energy
        nb_receiver_memory_writes_for_data = ceil(
            tensor.size / sender_top_memory_level.write_bw
        )
        receiver_energy = (
            receiver_top_memory_level.write_energy * nb_receiver_memory_writes_for_data
        )

        return sender_energy + receiver_energy

    def block_offchip_links(
        self, too_large_operands, core_id, start_timestep, duration, cn_id
    ) -> int:
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
        if "O" in too_large_operands:
            links_to_block.append(self.get_links_for_pair(core, offchip_core))
        if [op for op in too_large_operands if op != "O"]:
            links_to_block.append(self.get_links_for_pair(offchip_core, core))
        if not too_large_operands:
            return start_timestep
        # Get the worst case start time of all the links for all the operands
        worst_case_start_time = max(
            [link.available_from for links in links_to_block for link in links]
        )
        worst_case_start_time = max(start_timestep, worst_case_start_time)
        links_set = set((link for links in links_to_block for link in links))
        for link in links_set:
            blocking_start_timestep, blocking_end_timestep = link.block(
                worst_case_start_time, duration, cn_id
            )
            assert (
                blocking_start_timestep == worst_case_start_time
            ), "Mismatch between worst case link start time and effective link block start time."
        # for links in links_to_block:
        #     for link in links: # There can be multiple links if the offchip is not directly connected to this core
        #         blocking_start_timestep, blocking_end_timestep = link.block(worst_case_start_time, duration, cn_id)
        #         assert blocking_start_timestep == worst_case_start_time, "Mismatch between worst case link start time and effective link block start time."
        return worst_case_start_time

    def contains_tensor(self, tensor: Tensor, top_instance):
        return self.memory_manager.contains(tensor, top_instance)

    def find_tensor(self, tensor: Tensor):
        return self.memory_manager.find_tensor(tensor)

    def find_tensor_in_top_instances(self, tensor: Tensor):
        return self.memory_manager.find_tensor_in_top_instances(tensor)

    def has_shared_memory(self, core_id_a, core_id_b, mem_op_a, mem_op_b):
        """Check whether two cores have a shared top level memory instance for a given memory operand.

        Args:
            core_id_a (int): The first core id.
            core_id_b (int): The second core id.
            mem_op_a (str): The memory operand for the tensor in core a.
            mem_op_b (str): The memory operand for the tensor in core b.
        """
        core_a = self.get_core(core_id_a)
        core_b = self.get_core(core_id_b)
        top_memory_instance_a = next(
            (
                ml.memory_instance
                for ml, out_degree in core_a.memory_hierarchy.out_degree()
                if out_degree == 0 and mem_op_a in ml.operands
            )
        )
        top_memory_instance_b = next(
            (
                ml.memory_instance
                for ml, out_degree in core_b.memory_hierarchy.out_degree()
                if out_degree == 0 and mem_op_b in ml.operands
            )
        )
        if top_memory_instance_a is top_memory_instance_b:
            return True
        return False

    def get_top_instances_of_core(self, core_id):
        core = self.get_core(core_id)
        top_instances = self.memory_manager.top_instances[core]
        return top_instances

    def get_top_instance_of_core(self, core_id, mem_op):
        core = self.get_core(core_id)
        top_instances = self.memory_manager.top_instances[core]
        for instance in top_instances:
            core_idx = self.memory_manager.cores_per_top_instance[instance].index(core)
            instance_mem_ops = self.memory_manager.memory_operands_per_top_instance[
                instance
            ][core_idx]
            if mem_op in instance_mem_ops:
                return instance
        raise ValueError(
            f"No top instance for core {core_id} with memory operand {mem_op}."
        )
