from math import ceil
from typing import Any

from zigzag.datatypes import MemoryOperand
from zigzag.mapping.spatial_mapping import SpatialMapping
from zigzag.utils import DiGraphWrapper

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.tensor import SubviewTensor


class CoreGraph(DiGraphWrapper[Core]):
    """Represents the core structure of an accelerator"""


class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    In this Stream version, the cores are actually a graph with directed edges representing communication links.
    """

    def __init__(
        self,
        name: str,
        cores: CoreGraph,
        nb_shared_mem_groups: int,
        offchip_core_id: int | None = None,
    ):
        """ """
        self.name = name
        self.cores = cores
        self.offchip_core_id = offchip_core_id
        self.nb_shared_mem_groups = nb_shared_mem_groups

    def get_core(self, core_id: int) -> Core:
        """s
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        return self.cores.get_node_with_id(core_id)

    def get_offchip_core(self) -> Core:
        """Return the offchip core."""
        assert self.offchip_core_id, "This accelerator has no offchip core id."
        return self.get_core(self.offchip_core_id)

    def get_spatial_mapping_from_core(self, core_allocation: list[int]) -> SpatialMapping:
        """Iff the dataflows of all given cores is the same, return that dataflow. Otherwise, throw an error"""
        all_dataflows = [self.get_core(core_id).dataflows for core_id in core_allocation]
        some_dataflow = all_dataflows.pop()

        # All cores have same dataflow
        if some_dataflow is not None and all(some_dataflow == dataflow for dataflow in all_dataflows):
            return some_dataflow

        raise ValueError("Unclear which dataflow to return or no valid dataflow found.")

    def has_shared_memory(self, core_id_a: int, core_id_b: int, mem_op_a: MemoryOperand, mem_op_b: MemoryOperand):
        """Check whether two cores have a shared top level memory instance for a given memory operand.

        Args:
            core_id_a : The first core id.
            core_id_b : The second core id.
            mem_op_a : The memory operand for the tensor in core a.
            mem_op_b : The memory operand for the tensor in core b.
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

    def remove_all(
        self,
        core: Core,
        memory_operand: MemoryOperand,
        timestep: int,
        exceptions: list[SubviewTensor] | None = None,
        write_back_to_offchip: bool = False,
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
        if exceptions is None:
            exceptions = []
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        top_instance = self.get_top_instance_of_core(core, memory_operand)
        # stored_tensors = self.stored_tensors[core][top_level_idx]
        t = timestep
        for tensor in self.memory_manager.get_tensors_stored_at_timestep(top_instance, timestep):
            if tensor not in exceptions:
                t, link_energy_cost, memory_energy_cost = self.remove(
                    tensor, core, memory_operand, t, write_back_to_offchip
                )
                total_link_energy_cost += link_energy_cost
                total_memory_energy_cost += memory_energy_cost
        return t, total_link_energy_cost, total_memory_energy_cost

    def make_space_for(
        self,
        tensor: SubviewTensor,
        core: Core,
        memory_op: MemoryOperand,
        timestep: int,
        tensors_to_avoid_evicting: list[SubviewTensor] | None = None,
    ):
        """Make space for the given tensor on the given core by evicting already stored tensors if necessary.

        Args:
            tensor (SubviewTensor): The tensor to make space for.
            core (Core): The core where the tensor will be stored.
            memory_operand (str): The memory operand on the core.
            timestep (int): The timestep at which to make space for.
        """
        if tensors_to_avoid_evicting is None:
            tensors_to_avoid_evicting = []
        total_eviction_link_energy_cost = 0
        total_eviction_memory_energy_cost = 0

        top_instance = self.get_top_instance_of_core(core, memory_op)

        # If the tensor is already present in the core, no need to evict anything
        if self.memory_manager.contains(tensor, top_instance):
            return timestep, total_eviction_link_energy_cost, total_eviction_memory_energy_cost

        # Get the timestep at which there's enough space for this tensor
        enough_space_timestep = self.memory_manager.get_timestep_for_tensor_addition(
            tensor,
            core.id,
            timestep,
            memory_op=tensor.memory_operand,
        )

        tensors_to_evict = self.memory_manager.find_best_tensor_combination_to_evict_aie2(
            top_instance,
            tensor,
            enough_space_timestep,
            exceptions=tensors_to_avoid_evicting,
        )
        if core.id == self.offchip_core_id and tensors_to_evict:
            raise ValueError("Evictions required in offchip memory. Consider making offchip larger.")
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
        t_evictions_complete = max(enough_space_timestep, t_evictions_complete)
        return (
            t_evictions_complete,
            total_eviction_link_energy_cost,
            total_eviction_memory_energy_cost,
        )

    def transfer_tensor_to_core(
        self,
        tensor: SubviewTensor,
        receiving_core_id: int,
        tensor_operand: MemoryOperand,
        non_evictable_tensors: list[SubviewTensor],
        sending_core_id: int | None = None,
        transfer_bandwidth_fraction: float = 100.0,
    ) -> tuple[int, float, float, float, float, bool]:
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
        If one of the links already transferred the tensor, we broadcast if possible.

        Args:
            tensor (SubviewTensor): The tensor to transfer.
            receiving_core_id (int): The id of the core that needs to receive the tensor.
            tensor_operand (str): The memory operand where the tensor needs to be stored.
            non_evictable_tensors (list): the stored tensor that cannot be evicted
            sending_core_id (int, optional): The id of the core that should transfer the tensor.
        """
        ################################# STEP 0 #################################
        # Check if the tensor is already on the receiving core
        # Get the top instance where the tensor will be transferred to
        receiving_core = self.get_core(receiving_core_id)
        receiving_top_instance = self.get_top_instance_of_core(receiving_core_id, tensor_operand)
        if self.memory_manager.contains(tensor, receiving_top_instance):
            return -1, 0, 0, 0, 0, False
        ################################# STEP 1 #################################
        # Get the top instance storing the tensor
        # If a sending core id is provided, we get the instance of that core.
        # Else, we find the instance where the tensor has been stored the longest
        if sending_core_id is not None:
            storing_instance = self.get_top_instance_of_core(sending_core_id, tensor.memory_operand)
            assert self.contains_tensor(tensor, storing_instance)
            available_since_timestep = self.memory_manager.top_instance_available_since_timestep[storing_instance][
                tensor.equality_hash()
            ]
        else:
            (_, available_since_timesteps) = self.find_tensor_in_top_instances(tensor)
            # Pick the core that has stored the tensor the longest
            available_since_timestep = min(available_since_timesteps.values())
            storing_instance = next(
                top_instance
                for (top_instance, timestep) in available_since_timesteps.items()
                if timestep == available_since_timestep
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
        links = self.communication_manager.get_all_links_for_pair(sender_core, receiving_core)[0]
        links = {link: link.bandwidth for link in links}
        transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
        transfer_start = self.communication_manager.get_links_idle_window(
            links,
            evictions_complete_timestep,
            transfer_duration,
            {link: [tensor] for link in links},
        )
        transfer_end = transfer_start + transfer_duration
        ################################# STEP 5 #################################
        # Spawn the tensor on the receiving core
        self.spawn(tensor, receiving_core, tensor_operand, transfer_start, transfer_end)

        # Register transfer sending core -> receiving core
        (
            transfer_link_energy_cost,
            transfer_memory_energy_cost,
        ) = self.communication_manager.transfer_tensor(
            sender_core,
            receiving_core,
            tensor,
            tensor_operand,
            transfer_start,
            transfer_duration,
            link_bw_fraction=transfer_bandwidth_fraction,
        )

        # Remove from sender core (except if it is offchip)
        if sender_core.id != self.offchip_core_id:
            not_on_producing_core = sender_core.id != tensor.origin.chosen_core_allocation
            storing_instance = self.get_storing_memory_instance(tensor, sender_core)
            tensor_priority = tensor.get_instance_priority(storing_instance, self.memory_manager)
            if not_on_producing_core and tensor_priority == 0:
                self.remove_tensor(tensor, sender_core, memory_op=tensor.memory_operand, timestep=transfer_end)

        return transfer_link_energy_cost, transfer_memory_energy_cost

    def find_earliest_time_for_transfer(
        self,
        tensor: SubviewTensor,
        sending_core: Core,
        receiving_core: Core,
        earliest_t: int,
        bandwidth_fraction: float = 1,
    ):
        """Find the earliest time  >= `earliest_t` at which a tensor transfer between 2 cores can happen."""
        assert 0 < bandwidth_fraction <= 1
        windows: list[tuple[int, int]] = []

        links = self.communication_manager.get_all_links_for_pair(sending_core, receiving_core)[
            0
        ]  # Take the first path
        links_with_bw = {link: ceil(bandwidth_fraction * link.bandwidth) for link in links}
        start, end = self.find_transfer_start_and_end_time(tensor, links_with_bw, earliest_t)
        windows.append((start, end))

        ends = [end for _, end in windows]
        best_idx = ends.index(min(ends))
        best_window = windows[best_idx]
        return best_window

    def find_transfer_start_and_end_time(
        self, tensor: SubviewTensor, links_bw: dict[CommunicationLink, int], earliest_t: int
    ):
        """
        Given the links to transfer across and corresponding available bandwidths, return the earliest transfer start
        and end time for this tensor.

        Args:
            tensor: The tensor to transfer
            links_bw: link and corresponding transfer bandwidth
        """
        slowest_bw = min(links_bw.values())
        transfer_duration = ceil(tensor.size / slowest_bw)
        tensor_bw_per_link = {link: [(tensor, link_bw)] for link, link_bw in links_bw.items()}
        transfer_start = self.communication_manager.get_links_idle_window(
            tensor_bw_per_link=tensor_bw_per_link,
            start_timestep=earliest_t,
            duration=transfer_duration,
        )
        transfer_end = transfer_start + transfer_duration
        return transfer_start, transfer_end

    def get_memory_energy_cost_of_transfer(
        self,
        tensor: SubviewTensor,
        sender: Core | int,
        receiver: Core | int,
        sender_memory_operand: MemoryOperand,
        receiver_memory_operand: MemoryOperand,
    ):
        # Convert given sender and receiver to Core object if given as ids
        if isinstance(sender, int):
            sender = self.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.get_core(receiver)

        # Get the top level of output memory for the sender and the top level of input memory for the consumer_operand
        # Sender memory energy
        sender_top_instance = sender.get_top_memory_instance(sender_memory_operand)
        sender_bw_min = sender_top_instance.ports[0].bw_min
        sender_bw_max = sender_top_instance.ports[0].bw_max
        nb_sender_memory_reads_for_data = ceil(tensor.size / sender_bw_min)
        sender_energy = sender_top_instance.r_cost * (sender_bw_min / sender_bw_max) * nb_sender_memory_reads_for_data
        # Receiver memory energy
        receiver_top_instance = receiver.get_top_memory_instance(receiver_memory_operand)
        receiver_bw_min = receiver_top_instance.ports[0].bw_min
        receiver_bw_max = receiver_top_instance.ports[0].bw_max
        nb_receiver_memory_writes_for_data = ceil(tensor.size / receiver_bw_min)
        receiver_energy = (
            receiver_top_instance.w_cost * (receiver_bw_min / receiver_bw_max) * nb_receiver_memory_writes_for_data
        )

        return sender_energy + receiver_energy

    @property
    def core_list(self) -> list[Core]:
        return list(self.cores.node_list)

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self) -> dict[str, Any]:
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.cores}
