from itertools import combinations
import numpy as np

# from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.tensor import Tensor
import bisect
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Class that keeps track of the memory state of all top level memories of each core."""

    def __init__(self, accelerator) -> None:
        self.accelerator = accelerator
        # For each core in the accelerator, create a list containing the top level memories, instances, which memory operands they store and their capacity
        self.top_levels = {}  # top level memory of each core
        self.top_instances = {}
        self.memory_operands = {}  # memory operand stored by every top level memory
        for core_id, core in sorted(
            [(core.id, core) for core in self.accelerator.cores.nodes()]
        ):
            top_levels = list(
                (
                    level
                    for level, out_degree in core.memory_hierarchy.out_degree()
                    if out_degree == 0
                )
            )
            self.top_levels[core] = top_levels
            self.top_instances[core] = [level.memory_instance for level in top_levels]
            self.memory_operands[core] = [level.operands for level in top_levels]

        self.unique_top_instances = set()
        self.cores_per_top_instance = {}
        self.memory_operands_per_top_instance = {}
        # Some top level memories instances might be shared, thus we keep for each unique top memory instance the:
        # - capacity
        self.top_instance_capacities = {}
        # - avilable memory space
        self.top_instance_available = {}
        # - stored tensors
        self.top_instance_stored_tensors = {}
        # - timestep since when they have been stored
        self.top_instance_stored_since_timestep = {}
        # - the changes in memory usage through time
        self.top_instance_delta_history = {}
        # - the cumulative sum of stored tensors
        self.top_instance_stored_cumsum = {}
        # - the current timestep, i.e. when the top level was lastly used
        self.top_instance_current_timestep = {}
        for core, top_levels in self.top_levels.items():
            for top_level in top_levels:
                top_instance = top_level.memory_instance
                if top_instance not in self.unique_top_instances:
                    self.unique_top_instances.add(top_instance)
                    self.cores_per_top_instance[top_instance] = [core]
                    self.memory_operands_per_top_instance[top_instance] = [
                        tuple(top_level.operands)
                    ]
                    self.top_instance_capacities[top_instance] = top_instance.size
                    self.top_instance_available[top_instance] = top_instance.size
                    self.top_instance_stored_tensors[top_instance] = []
                    self.top_instance_stored_since_timestep[top_instance] = {}
                    self.top_instance_delta_history[top_instance] = [[0, 0]]
                    self.top_instance_stored_cumsum[top_instance] = np.array([[0, 0]])
                    self.top_instance_current_timestep[top_instance] = 0
                else:
                    self.cores_per_top_instance[top_instance].append(core)
                    self.memory_operands_per_top_instance[top_instance].append(
                        tuple(top_level.operands)
                    )

        self.offchip_core_id = self.accelerator.offchip_core_id

    def contains(self, tensor: Tensor, top_instance):
        return any(
            [
                tensor.equality_hash() == stored_tensor.equality_hash()
                for stored_tensor in self.top_instance_stored_tensors[top_instance]
            ]
        )

    def find_tensor_in_top_instances(self, tensor: Tensor):
        """Find the top memory instances that are storing this tensor.

        Args:
            tensor (Tensor): _description_
        """
        equality_hash = tensor.equality_hash()
        # Find all instances storing this tensor
        instances_storing_tensor = set()
        stored_since_timesteps = {}
        for top_instance, stored_tensors in self.top_instance_stored_tensors.items():
            if any(
                (
                    equality_hash == stored_tensor.equality_hash()
                    for stored_tensor in stored_tensors
                )
            ):
                instances_storing_tensor.add(top_instance)
                stored_since_timesteps[
                    top_instance
                ] = self.top_instance_stored_since_timestep[top_instance][equality_hash]
        # If no instances are storing this tensor, raise error
        if not instances_storing_tensor:
            raise ValueError(f"Tensor {tensor} was not found in any of the instances.")
        return instances_storing_tensor, stored_since_timesteps

    def find_tensor(self, tensor: Tensor):
        (
            instances_storing_tensor,
            stored_since_timesteps,
        ) = self.find_tensor_in_top_instances(tensor)
        cores_storing_tensor = []
        top_instance_idxs = []
        stored_since = []
        # Find which cores have these instances as their top instance
        for core, top_instances in self.top_instances.items():
            for top_instance_idx, top_instance in enumerate(top_instances):
                if top_instance in instances_storing_tensor:
                    cores_storing_tensor.append(core.id)
                    top_instance_idxs.append(top_instance_idx)
                    stored_since.append(stored_since_timesteps[top_instance])
                    # Remove the entry so that next cores that have this shared memory don't get added
                    instances_storing_tensor.remove(top_instance)
                    del stored_since_timesteps[top_instance]

        return cores_storing_tensor, top_instance_idxs, stored_since

    def add_tensor_to_core(
        self,
        tensor: Tensor,
        core_id: int,
        timestep: int,
        timestep_end: int,
        tensors_to_avoid_evicting: list,
        memory_op: str = None,
    ):
        timestep_delta = timestep_end - timestep
        total_eviction_link_energy_cost = 0
        total_eviction_memory_energy_cost = 0
        core = self.accelerator.get_core(core_id)
        tensor_size = tensor.size
        if not memory_op:
            memory_op = tensor.memory_operand
        top_level_idx = self.get_top_level_idx(core, memory_op)
        top_instance = self.top_instances[core][top_level_idx]

        if self.contains(tensor, top_instance):
            return (
                timestep,
                total_eviction_link_energy_cost,
                total_eviction_memory_energy_cost,
            )

        ## Get the tensors that were stored at this timestep.
        # Because of shared memory there might be tensors that don't exist yet
        # but are already in the top_instance_stored_tensors because of
        # the scheduler's non-causal behavior.
        stored_tensors = self.top_instance_stored_tensors[top_instance]
        stored_tensors = [
            stored_tensor
            for stored_tensor in stored_tensors
            if self.top_instance_stored_since_timestep[top_instance][
                stored_tensor.equality_hash()
            ]
            <= timestep
        ]

        # If there is no equivalent tensor in the core, remove tensors until we have enough space
        # Tensors are removed based on their priority value
        memory_capacity = self.top_instance_capacities[top_instance]
        tensors_to_evict = self.find_best_tensor_combination_to_evict_fast(
            core_id, tensor, stored_tensors, memory_capacity, tensors_to_avoid_evicting
        )
        for tensor_to_evict in tensors_to_evict:
            (
                end_of_eviction_timestep,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
            ) = self.remove_tensor_from_core(
                core,
                top_level_idx,
                tensor_to_evict,
                timestep,
                write_back_to_offchip=True,
            )
            if end_of_eviction_timestep > timestep:
                timestep = end_of_eviction_timestep
            total_eviction_link_energy_cost += eviction_link_energy_cost
            total_eviction_memory_energy_cost += eviction_memory_energy_cost

        # Now that we have enough space, we add this tensor
        self.top_instance_stored_tensors[top_instance].append(tensor)
        self.top_instance_stored_since_timestep[top_instance][
            tensor.equality_hash()
        ] = (timestep + timestep_delta)
        self.top_instance_available[top_instance] -= tensor_size
        # use package bisect to insert the [timestep, tensor_size] in the correct timeframe, enable data preloading
        bisect.insort(
            self.top_instance_delta_history[top_instance], [timestep, tensor_size]
        )

        # Use numpy searchsorted to find the where the timestep should be inserted
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        insert_idx = np.searchsorted(all_timesteps, timestep)
        timestep_already_present = (
            insert_idx < len(all_timesteps) and all_timesteps[insert_idx] == timestep
        )

        # We first update the remaining usages of later timesteps
        # If timestep was already in all_timesteps, this timestep will also be updated
        relevant_usages = all_usages[insert_idx:]
        updated_relevant_usages = relevant_usages + tensor.size
        self.top_instance_stored_cumsum[top_instance][
            insert_idx:, 1
        ] = updated_relevant_usages

        # If the timestep was not in all_timesteps, it will be inserted here
        if not timestep_already_present:
            self.top_instance_stored_cumsum[top_instance] = np.insert(
                self.top_instance_stored_cumsum[top_instance],
                insert_idx,
                [timestep, all_usages[insert_idx - 1] + tensor.size],
                axis=0,
            )

        return (
            timestep,
            total_eviction_link_energy_cost,
            total_eviction_memory_energy_cost,
        )

    def test_add_tensor_to_core(
        self,
        tensor: Tensor,
        core_id: int,
        test_timestep: int,
        worst_case_timestep: int,
        data_transfer_duration: int,
        memory_op: str,
    ) -> int:
        """
        This function gives the earliest timestep since test_timestep that the tensor can be added to the core.
        A timestep is picked as close as possible to the actual computatino (worst_case_timestep) that doesn't block the computation

        Args:
        tensor (Tensor): The tensor to be added to the core.
        core_id (int): The core id that is going to receive the tensor.
        test_timestep (int): The timestep from which to start considering make this tensor data transfer.
        data_transfer_duration (int): The duration of the transfer.
        memory_op (str): The memory operand storing the tensor on the receiving end of the transfer.
        worst_case_timestep (int): when the data cannot be prefetched (no enough space), the latest timestep that it needs to be transferred.

        Returns:
        can_transfer_from_timestep (int): The earliest timestep at which the transfer can actually start.
        """
        ideal_timestep = max(
            test_timestep, worst_case_timestep - data_transfer_duration
        )
        core = self.accelerator.get_core(core_id)
        top_level_idx = self.get_top_level_idx(core, memory_op)
        top_instance = self.top_instances[core][top_level_idx]
        top_instance_capacity = self.top_instance_capacities[top_instance]
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        first_possible_idx = np.searchsorted(all_timesteps, ideal_timestep)
        last_possible_idx = len(all_timesteps)
        for possible_idx in range(first_possible_idx - 1, last_possible_idx):
            relevant_usages = all_usages[possible_idx:]
            updated_relevant_usages = relevant_usages + tensor.size
            if max(updated_relevant_usages) <= top_instance_capacity:
                can_transfer_from_timestep = max(
                    all_timesteps[possible_idx], ideal_timestep
                )
                return can_transfer_from_timestep
        # If we can't add it at any timestep, raise error
        return worst_case_timestep

    def generate_all_combinations(self, lst):
        for i in range(1, len(lst) + 1):
            for comb in combinations(lst, i):
                yield comb

    def find_best_tensor_combination_to_evict(
        self,
        top_instance,
        tensor_to_add,
        stored_tensors,
        capacity,
        tensors_to_avoid_evicting,
    ):
        relevant_tensors_to_avoid_evicting = [
            tensor for tensor in tensors_to_avoid_evicting if tensor in stored_tensors
        ]
        stored_tensors_size = sum(
            (stored_tensor.size for stored_tensor in stored_tensors)
        )
        if stored_tensors_size + tensor_to_add.size <= capacity:
            return []
        min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
        min_score, best_combination_to_evict = float("inf"), []
        for combination in self.generate_all_combinations(
            [
                tensor
                for tensor in stored_tensors
                if tensor not in relevant_tensors_to_avoid_evicting
            ]
        ):
            score = sum(
                (
                    stored_tensor.instance_priorities[top_instance] * stored_tensor.size
                    for stored_tensor in combination
                )
            )
            evicted_size = sum((stored_tensor.size for stored_tensor in combination))
            if evicted_size >= min_size_to_evict and score < min_score:
                min_score = score
                best_combination_to_evict = list(combination)
        if not best_combination_to_evict:
            raise ValueError(
                "The best tensor combination to evict is empty. tensors_to_avoid_evicting might be too large for the candidate."
            )
        return best_combination_to_evict

    def find_best_tensor_combination_to_evict_fast(
        self,
        top_instance,
        tensor_to_add,
        stored_tensors,
        capacity,
        tensors_to_avoid_evicting,
    ):
        relevant_tensors_to_avoid_evicting = [
            tensor for tensor in tensors_to_avoid_evicting if tensor in stored_tensors
        ]
        stored_tensors_size = sum(
            (stored_tensor.size for stored_tensor in stored_tensors)
        )
        min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
        if (
            min_size_to_evict < 0
        ):  # no need to evict any tensor, the memory's space is enough
            return []
        evictable_tensors = [
            tensor
            for tensor in stored_tensors
            if tensor not in relevant_tensors_to_avoid_evicting
        ]
        evictable_tensors_priority_size = []
        for tensor in evictable_tensors:
            instance_priority = tensor.get_instance_priority(top_instance, self)
            importance = instance_priority * tensor.size
            evictable_tensors_priority_size.append(importance)

        if not evictable_tensors:
            evictable_tensors_priority_size, evictable_tensors = [], []
        else:
            evictable_tensors_priority_size, evictable_tensors = zip(
                *sorted(zip(evictable_tensors_priority_size, evictable_tensors))
            )  # sort them
        evictable_tensors_size = [tensor.size for tensor in evictable_tensors]
        evictable_tensors_size_sums = [
            sum(evictable_tensors_size[:i])
            for i in range(0, len(evictable_tensors_size) + 1)
        ]
        try:
            idx_satisfying_min_size_to_evict = next(
                (
                    i
                    for i, size_sum in enumerate(evictable_tensors_size_sums)
                    if size_sum >= min_size_to_evict
                )
            )
        except StopIteration:
            raise ValueError(
                f"The evictable tensors {evictable_tensors} and their sizes {evictable_tensors_size} are too small to evict a size of {min_size_to_evict}."
            )
        tensors_to_evict = evictable_tensors[:idx_satisfying_min_size_to_evict]
        return tensors_to_evict

    def remove_tensor_from_core(
        self,
        core,
        top_level_idx,
        tensor: Tensor,
        timestep: int,
        write_back_to_offchip: bool = True,
    ):
        tensor_size = tensor.size

        # Transfer the tensor to off-chip if it's not present there
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        offchip_instance = self.accelerator.get_top_instance_of_core(
            self.offchip_core_id, tensor.memory_operand
        )
        should_be_written_to_offchip = write_back_to_offchip and not self.contains(
            tensor, offchip_instance
        )
        # current_timestep = max(timestep, self.current_timestep[core][top_level_idx])
        current_timestep = timestep
        if should_be_written_to_offchip:
            (
                transfer_start,
                transfer_end,
                link_energy_cost,
                memory_energy_cost,
            ) = self.accelerator.transfer_data(
                tensor,
                core,
                self.offchip_core_id,
                tensor.memory_operand,
                current_timestep,
            )
            current_timestep = transfer_end
            total_link_energy_cost += link_energy_cost
            total_memory_energy_cost += memory_energy_cost
            self.add_tensor_to_core(
                tensor, self.offchip_core_id, transfer_start, transfer_end, []
            )  # no tensors to avoid evicting on offchip core
            # self.current_timestep[core][top_level_idx] = current_timestep

        try:
            top_instance = self.top_instances[core][top_level_idx]
            equivalent_tensor = next(
                (
                    stored_tensor
                    for stored_tensor in self.top_instance_stored_tensors[top_instance]
                    if stored_tensor.equality_hash() == tensor.equality_hash()
                )
            )
        except StopIteration:
            raise ValueError(
                f"No tensor found equal to {tensor} in core {core} top_level_idx {top_level_idx}."
            )
        self.top_instance_stored_tensors[top_instance].remove(equivalent_tensor)
        del self.top_instance_stored_since_timestep[top_instance][
            tensor.equality_hash()
        ]

        self.top_instance_available[top_instance] += tensor_size
        bisect.insort(
            self.top_instance_delta_history[top_instance],
            [current_timestep, -tensor_size],
        )

        # Use numpy searchsorted to find the where the current_timestep should be inserted
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        insert_idx = np.searchsorted(all_timesteps, current_timestep)
        timestep_already_present = (
            insert_idx < len(all_timesteps)
            and all_timesteps[insert_idx] == current_timestep
        )

        # We first update the remaining usages of later timesteps
        # If timestep was already in all_timesteps, this timestep will also be updated
        relevant_usages = all_usages[insert_idx:]
        updated_relevant_usages = relevant_usages - tensor.size
        self.top_instance_stored_cumsum[top_instance][
            insert_idx:, 1
        ] = updated_relevant_usages

        # If the timestep was not in all_timesteps, it will be inserted here
        if not timestep_already_present:
            self.top_instance_stored_cumsum[top_instance] = np.insert(
                self.top_instance_stored_cumsum[top_instance],
                insert_idx,
                [current_timestep, all_usages[insert_idx - 1] - tensor.size],
                axis=0,
            )

        return current_timestep, total_link_energy_cost, total_memory_energy_cost

    def remove_tensor_from_top_instance(
        self,
        top_instance,
        tensor: Tensor,
        timestep: int,
        write_back_to_offchip: bool = True,
    ):
        # Get the core of this top instance.
        # If there's more than one, pick the first one.
        core = self.cores_per_top_instance[top_instance][0]

        tensor_size = tensor.size

        # Transfer the tensor to off-chip if it's not present there
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        offchip_instance = self.accelerator.get_top_instance_of_core(
            self.offchip_core_id, tensor.memory_operand
        )
        should_be_written_to_offchip = write_back_to_offchip and not self.contains(
            tensor, offchip_instance
        )
        # current_timestep = max(timestep, self.current_timestep[core][top_level_idx])
        current_timestep = timestep
        if should_be_written_to_offchip:
            (
                transfer_start,
                transfer_end,
                link_energy_cost,
                memory_energy_cost,
            ) = self.accelerator.transfer_data(
                tensor,
                core,
                self.offchip_core_id,
                tensor.memory_operand,
                current_timestep,
            )
            current_timestep = transfer_end
            total_link_energy_cost += link_energy_cost
            total_memory_energy_cost += memory_energy_cost
            self.add_tensor_to_core(
                tensor, self.offchip_core_id, transfer_start, transfer_end, []
            )  # no tensors to avoid evicting on offchip core
            # self.current_timestep[core][top_level_idx] = current_timestep

        try:
            equivalent_tensor = next(
                (
                    stored_tensor
                    for stored_tensor in self.top_instance_stored_tensors[top_instance]
                    if stored_tensor.equality_hash() == tensor.equality_hash()
                )
            )
        except StopIteration:
            raise ValueError(
                f"No tensor found equal to {tensor} in top memory instance {top_instance}."
            )
        self.top_instance_stored_tensors[top_instance].remove(equivalent_tensor)
        del self.top_instance_stored_since_timestep[top_instance][
            tensor.equality_hash()
        ]

        self.top_instance_available[top_instance] += tensor_size
        bisect.insort(
            self.top_instance_delta_history[top_instance],
            [current_timestep, -tensor_size],
        )

        # Use numpy searchsorted to find the where the current_timestep should be inserted
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        insert_idx = np.searchsorted(all_timesteps, current_timestep)
        timestep_already_present = (
            insert_idx < len(all_timesteps)
            and all_timesteps[insert_idx] == current_timestep
        )

        # We first update the remaining usages of later timesteps
        # If timestep was already in all_timesteps, this timestep will also be updated
        relevant_usages = all_usages[insert_idx:]
        updated_relevant_usages = relevant_usages - tensor.size
        self.top_instance_stored_cumsum[top_instance][
            insert_idx:, 1
        ] = updated_relevant_usages

        # If the timestep was not in all_timesteps, it will be inserted here
        if not timestep_already_present:
            self.top_instance_stored_cumsum[top_instance] = np.insert(
                self.top_instance_stored_cumsum[top_instance],
                insert_idx,
                [current_timestep, all_usages[insert_idx - 1] - tensor.size],
                axis=0,
            )

        return current_timestep, total_link_energy_cost, total_memory_energy_cost

    def evict_all_tensors_from_core(
        self, core_id, memory_operand, timestep, tensors_to_avoid_evicting
    ):
        """Evict all stored tensors from core's memory storing memory_operand.
        All non-constant tensors that are not present in the off-chip memory will be written to the off-chip memory.

        Args:
            core_id (int): The id of the core to evict all tensors from.
            memory_operand (str): The memory operand for which we want to evict all tensors.
            timestep (int): The timestep at which the tensors should be removed.
            tensors_to_avoid_evicting (list): List of tensors that shouldn't be evicted (because they are needed for this CN's execution).
        """
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        core = self.accelerator.get_core(core_id)
        top_level_idx = self.get_top_level_idx(core, memory_operand)
        top_instance = self.top_instances[core][top_level_idx]
        # stored_tensors = self.stored_tensors[core][top_level_idx]
        for tensor in self.top_instance_stored_tensors[top_instance].copy():
            if tensor in tensors_to_avoid_evicting:
                continue  # Skip evicting this tensor (e.g. it's an input for this candidate)
            (
                timestep,
                link_energy_cost,
                memory_energy_cost,
            ) = self.remove_tensor_from_core(
                core, top_level_idx, tensor, timestep, write_back_to_offchip=True
            )
            total_link_energy_cost += link_energy_cost
            total_memory_energy_cost += memory_energy_cost
        return total_link_energy_cost, total_memory_energy_cost, timestep

    def get_top_level_idx(self, core, memory_operand):
        """Return the index of the top memory that stores memory_operand, index referring to the order in which they are stored in the list for this core

        Args:
            core (_type_): _description_
            memory_operand (_type_): _description_
        """
        return next(
            (
                idx
                for idx, operands_top_level in enumerate(self.memory_operands[core])
                if memory_operand in operands_top_level
            )
        )
