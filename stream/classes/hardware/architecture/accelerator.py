from math import ceil
from networkx import DiGraph

from zigzag.classes.hardware.architecture.core import Core
from stream.classes.cost_model.memory_manager import MemoryManager
from stream.classes.cost_model.communication_manager import CommunicationManager
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
        self.memory_manager = MemoryManager(self)
        self.communication_manager = CommunicationManager(self)

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

    def spawn(
        self,
        tensor: Tensor,
        core: Core,
        memory_op: str,
        initial_timestep: int,
        available_timestep: int,
    ):
        """Spawns a tensor on a core.

        Args:
            tensor (Tensor): The tensor to be spawned.
            core (Core): The core on which to spawn the tensor.
            memory_op (str): The memory operand on the core where the tensor will spawn.
            initial_timestep (int): The timestep at which space will be reserved for the tensor.
            available_timestep (int): The timestep at which the tensor will become available. Different from initial_timestep when it is transferred.
        """
        self.memory_manager.add_tensor_to_core(
            tensor, core, initial_timestep, available_timestep, memory_op
        )

    def remove(self, tensor, core, memory_op, timestep, write_back_to_offchip=False):
        """Remove tensor from core. If required, transfer to offchip before removal.

        Args:
            tensor (Tensor): The tensor to remove.
            core (Core): The Core to remove the tensor from.
            memory_op (str): The memory operand of the tensor.
            timestep (int): The timestep to remove the tensor at.
            write_back_to_offchip (bool, optional): Write the tensor to offchip before removal. Defaults to False.
        """

        ################################# STEP 1 #################################
        # Transfer the tensor to off-chip if required and not present there
        link_energy_cost = 0
        memory_energy_cost = 0
        offchip_instance = self.get_top_instance_of_core(
            self.offchip_core_id, memory_op
        )
        should_be_written_to_offchip = (
            write_back_to_offchip and not self.contains_tensor(tensor, offchip_instance)
        )
        current_timestep = timestep
        if should_be_written_to_offchip:
            (
                transfer_end,
                transfer_link_energy_cost,
                transfer_memory_energy_cost,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
                came_from_offchip,
            ) = self.transfer_tensor_to_core(
                tensor,
                self.offchip_core_id,
                memory_op,
                non_evictable_tensors=[],
                sending_core_id=core.id,
            )
            # There should be no evictions as we are writing to offchip
            assert eviction_link_energy_cost == 0
            assert eviction_memory_energy_cost == 0
            assert not came_from_offchip
            link_energy_cost = transfer_link_energy_cost
            memory_energy_cost = transfer_memory_energy_cost
            current_timestep = max(current_timestep, transfer_end)

        ################################# STEP 2 #################################
        # Remove the tensor from the memory manager's attributes
        top_instance = self.get_top_instance_of_core(core, memory_op)
        self.memory_manager.remove_tensor_from_top_instance(
            top_instance,
            tensor,
            timestep,
        )

        return current_timestep, link_energy_cost, memory_energy_cost

    def remove_all(
        self, core, memory_operand, timestep, exceptions=[], write_back_to_offchip=False
    ):
        """Remove all tensors from a core's memory with the given memory operand.
        If required, the tensors are written back to offchip before removal.

        Args:
            core (Core): The Core to remove the tensor from
            memory_operand (str): The memory operand for which all tensors should be evicted.
            timestep (int): The timestep to remove the tensor at.
            exceptions (list): A list of tensors that should not be evicted.
            write_back_to_offchip (bool, optional): Write the tensor to offchip before removal. Defaults to False.
        """
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        top_instance = self.get_top_instance_of_core(core, memory_operand)
        # stored_tensors = self.stored_tensors[core][top_level_idx]
        t = timestep
        for tensor in self.memory_manager.get_tensors_stored_at_timestep(
            top_instance, timestep
        ):
            if not tensor in exceptions:
                t, link_energy_cost, memory_energy_cost = self.remove(
                    tensor, core, memory_operand, t, write_back_to_offchip
                )
                total_link_energy_cost += link_energy_cost
                total_memory_energy_cost += memory_energy_cost
        return t, total_link_energy_cost, total_memory_energy_cost

    def make_space_for(
        self,
        tensor: Tensor,
        core: Core,
        memory_op: str,
        timestep: int,
        tensors_to_avoid_evicting: list = [],
    ):
        """Make space for the given tensor on the given core by evicting already stored tensors if necessary.

        Args:
            tensor (Tensor): The tensor to make space for.
            core (Core): The core where the tensor will be stored.
            memory_operand (str): The memory operand on the core.
            timestep (int): The timestep at which to make space for.
        """
        total_eviction_link_energy_cost = 0
        total_eviction_memory_energy_cost = 0

        top_instance = self.get_top_instance_of_core(core, memory_op)

        tensors_to_evict = (
            self.memory_manager.find_best_tensor_combination_to_evict_fast(
                top_instance,
                tensor,
                timestep,
                exceptions=tensors_to_avoid_evicting,
            )
        )
        if core.id == self.offchip_core_id and tensors_to_evict:
            raise ValueError(
                "Evictions required in offchip memory. Consider making offchip larger."
            )
        t_evictions_complete = timestep
        for tensor_to_evict in tensors_to_evict:
            (
                t_eviction_complete,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
            ) = self.remove(
                tensor_to_evict,
                core,
                memory_op,
                timestep,
                write_back_to_offchip=True,
            )
            t_evictions_complete = max(t_evictions_complete, t_eviction_complete)
            total_eviction_link_energy_cost += eviction_link_energy_cost
            total_eviction_memory_energy_cost += eviction_memory_energy_cost
        return t_evictions_complete, total_eviction_link_energy_cost, total_eviction_memory_energy_cost

    def transfer_tensor_to_core(
        self,
        tensor: Tensor,
        receiving_core_id: int,
        tensor_operand: str,
        non_evictable_tensors: list,
        sending_core_id: int = None,
    ):
        """
        Transfer a tensor to a given core id.
        If the tensor is already present on the receiving core, nothing happens.

        This function computes when the transfer can take place based on three factors:
        1) The tensor is available for transfer on a sender core.
        2) The receiver core has enough space to store the tensor.
        3) The links between sender and receiver have a long enough idle window.

        TODO: The transfer is scheduled as close as possible to the computation

        The tensor is then added to the memory. Evictions are still possible if
        there wasn't enough space on the receiver core at any earlier timestep.

        Args:
            tensor (Tensor): The tensor to transfer.
            receiving_core_id (int): The id of the core that needs to receive the tensor.
            tensor_operand (str): The memory operand where the tensor needs to be stored.
            non_evictable_tensors (list): the stored tensor that cannot be evicted
            sending_core_id (int, optional): The id of the core that should transfer the tensor.
        """
        ################################# STEP 0 #################################
        # Check if the tensor is already on the receiving core
        # Get the top instance where the tensor will be transferred to
        receiving_core = self.get_core(receiving_core_id)
        receiving_top_instance = self.get_top_instance_of_core(
            receiving_core_id, tensor_operand
        )
        if self.memory_manager.contains(tensor, receiving_top_instance):
            return -1, 0, 0, 0, 0, False
        ################################# STEP 1 #################################
        # Get the top instance storing the tensor
        # If a sending core id is provided, we get the instance of that core.
        # Else, we find the instance where the tensor has been stored the longest
        if sending_core_id is not None:
            storing_instance = self.get_top_instance_of_core(
                sending_core_id, tensor.memory_operand
            )
            assert self.contains_tensor(tensor, storing_instance)
            available_since_timestep = (
                self.memory_manager.top_instance_available_since_timestep[
                    storing_instance
                ][tensor.equality_hash()]
            )
        else:
            (
                instances_storing_tensor,
                available_since_timesteps,
            ) = self.find_tensor_in_top_instances(tensor)
            # Pick the core that has stored the tensor the longest
            available_since_timestep = min(available_since_timesteps.values())
            storing_instance = next(
                (
                    top_instance
                    for (top_instance, timestep) in available_since_timesteps.items()
                    if timestep == available_since_timestep
                )
            )
        ################################# STEP 2 #################################
        # The receiver core has enough space to store the tensor.
        enough_space_timestep = self.memory_manager.get_timestep_for_tensor_addition(
            tensor,
            receiving_core_id,
            available_since_timestep,
            memory_op=tensor_operand,
        )
        ################################# STEP 3 #################################
        # Make space on the receiving core by evicting tensors if there was never enough space.
        (
            evictions_complete_timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = self.make_space_for(
            tensor,
            receiving_core,
            tensor_operand,
            enough_space_timestep,
            non_evictable_tensors,
        )
        ################################# STEP 4 #################################
        # The links between sender and receiver have a long enough idle window.
        sender_cores = self.memory_manager.cores_per_top_instance[storing_instance]
        # TODO If the storing_instance is a shared instance across more than one core,
        # TODO there will be multiple possible cores to transfer between.
        # TODO For now, we take the first one
        sender_core = sender_cores[0]
        links = self.communication_manager.get_links_for_pair(
            sender_core, receiving_core
        )
        transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
        transfer_start = self.communication_manager.get_links_idle_window(
            links, evictions_complete_timestep, transfer_duration
        )
        transfer_end = transfer_start + transfer_duration
        ################################# STEP 5 #################################
        # Spawn the tensor on the receiving core
        self.spawn(tensor, receiving_core, tensor_operand, transfer_start, transfer_end)
        ################################# STEP 6 #################################
        # Update the links involved in the communication and get the transfer energy cost
        (
            transfer_link_energy_cost,
            transfer_memory_energy_cost,
        ) = self.communication_manager.update_links(
            tensor,
            sender_core.id,
            receiving_core_id,
            tensor_operand,
            transfer_start,
            transfer_duration,
        )
        ################################# STEP 7 #################################
        # Remove the transfered tensor from the sender core (excluding DRAM)
        # if it is no longer needed.
        if sender_core.id == self.offchip_core_id:
            pass
        else:
            if (storing_instance not in tensor.instance_priorities) or (
                tensor.instance_priorities[storing_instance] == 0
            ):
                self.remove(
                    tensor,
                    sender_core,
                    tensor.memory_operand,
                    transfer_end,
                    write_back_to_offchip=False,
                )
        ################################# STEP 8 #################################
        # Give back flag that signals if the tensor came from offchip
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
        self, too_large_operands, core_id, start_timestep, duration, cn
    ) -> int:
        return self.communication_manager.block_offchip_links(
            too_large_operands, core_id, start_timestep, duration, cn
        )

    def contains_tensor(self, tensor: Tensor, top_instance):
        if isinstance(top_instance, int):  # assume core id
            memory_op = tensor.memory_operand
            top_instance = self.get_top_instance_of_core(top_instance, memory_op)

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
        return top_memory_instance_a is top_memory_instance_b

    def get_top_instances_of_core(self, core_id):
        core = self.get_core(core_id)
        top_instances = self.memory_manager.top_instances[core]
        return top_instances

    def get_top_instance_of_core(self, core, mem_op):
        if isinstance(core, int):
            core = self.get_core(core)
        top_instances = self.memory_manager.top_instances[core]
        for instance in top_instances:
            core_idx = self.memory_manager.cores_per_top_instance[instance].index(core)
            instance_mem_ops = self.memory_manager.memory_operands_per_top_instance[
                instance
            ][core_idx]
            if mem_op in instance_mem_ops:
                return instance
        raise ValueError(f"No top instance for {core} with memory operand {mem_op}.")
