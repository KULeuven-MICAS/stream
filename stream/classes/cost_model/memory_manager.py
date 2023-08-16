from itertools import combinations
import numpy as np
import bisect
import logging

from zigzag.classes.hardware.architecture.core import Core
from stream.classes.workload.tensor import Tensor

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
        # - timestep since when they have been available
        self.top_instance_available_since_timestep = {}
        # - the changes in memory usage through time
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
                    self.top_instance_available_since_timestep[top_instance] = {}
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
        available_since_timesteps = {}
        for top_instance, stored_tensors in self.top_instance_stored_tensors.items():
            if any(
                (
                    equality_hash == stored_tensor.equality_hash()
                    for stored_tensor in stored_tensors
                )
            ):
                instances_storing_tensor.add(top_instance)
                available_since_timesteps[
                    top_instance
                ] = self.top_instance_available_since_timestep[top_instance][equality_hash]
        # If no instances are storing this tensor, raise error
        if not instances_storing_tensor:
            raise ValueError(f"Tensor {tensor} was not found in any of the instances.")
        return instances_storing_tensor, available_since_timesteps

    def find_tensor(self, tensor: Tensor):
        (
            instances_storing_tensor,
            available_since_timesteps,
        ) = self.find_tensor_in_top_instances(tensor)
        cores_storing_tensor = []
        top_instance_idxs = []
        available_since = []
        # Find which cores have these instances as their top instance
        for core, top_instances in self.top_instances.items():
            for top_instance_idx, top_instance in enumerate(top_instances):
                if top_instance in instances_storing_tensor:
                    cores_storing_tensor.append(core.id)
                    top_instance_idxs.append(top_instance_idx)
                    available_since.append(available_since_timesteps[top_instance])
                    # Remove the entry so that next cores that have this shared memory don't get added
                    instances_storing_tensor.remove(top_instance)
                    del available_since_timesteps[top_instance]

        return cores_storing_tensor, top_instance_idxs, available_since

    def add_tensor_to_core(
        self,
        tensor: Tensor,
        core: Core,
        timestep: int,
        timestep_end: int,
        memory_op: str = None,
    ):
        """Add the tensor to the relevant memory manager attributes.
        This function does not handle evictions.
        An error is raised if there is not enough space to add it.

        Args:
            tensor (Tensor): The tensor to be added.
            core (Core): The core to add it to.
            timestep (int): The timestep at which space should be reserved for the tensor.
            timestep_end (int): The timestep at which the tensor is available.
            memory_op (str, optional): The memory operand where the tensor will be stored. Defaults to None.
        """

        if not memory_op:
            memory_op = tensor.memory_operand
        top_level_idx = self.get_top_level_idx(core, memory_op)
        top_instance = self.top_instances[core][top_level_idx]

        # Check if the tensor is already present
        if self.contains(tensor, top_instance):
            return

        # Add the tensor
        self.top_instance_stored_tensors[top_instance].append(tensor)
        self.top_instance_stored_since_timestep[top_instance][
            tensor.equality_hash()
        ] = timestep
        self.top_instance_available_since_timestep[top_instance][
            tensor.equality_hash()
        ] = timestep_end
        self.top_instance_available[top_instance] -= tensor.size

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
        if (
            np.max(updated_relevant_usages, initial=0)
            > self.top_instance_capacities[top_instance]
        ):
            raise ValueError(
                f"Inserting {tensor} in {top_instance} caused memory overflow."
            )
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

        return

    def get_timestep_for_tensor_addition(
        self,
        tensor: Tensor,
        core_id: int,
        timestep: int,
        memory_op: str,
    ) -> int:
        """
        This function gives the earliest timestep since 'timestep' that the tensor can be added to the core.
        If there is never enough space, the latest timestep is returned.

        Args:
        tensor (Tensor): The tensor to be added to the core.
        core_id (int): The core id that is going to receive the tensor.
        timestep (int): The timestep from which to start considering make this tensor data transfer.
        memory_op (str): The memory operand storing the tensor on the receiving end of the transfer.

        Returns:
        can_add_from_timestep (int): The earliest timestep at which the transfer can actually start.
        """
        core = self.accelerator.get_core(core_id)
        top_level_idx = self.get_top_level_idx(core, memory_op)
        top_instance = self.top_instances[core][top_level_idx]
        top_instance_capacity = self.top_instance_capacities[top_instance]
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        relevant_start_idx = np.searchsorted(all_timesteps, timestep, 'right') - 1
        if relevant_start_idx == len(all_timesteps):
            return timestep
        relevant_timesteps = all_timesteps[relevant_start_idx:]
        relevant_usages = all_usages[relevant_start_idx:]
        relevant_usages_reversed = relevant_usages[::-1]
        max_usage = np.max(relevant_usages_reversed)
        last_max_usage_idx = (
            len(relevant_usages_reversed) - np.argmax(relevant_usages_reversed) - 1
        )
        # abs_last_max_usage_idx = relevant_start_idx + last_max_usage_idx
        if max_usage + tensor.size <= top_instance_capacity:
            can_add_from_timestep = timestep
            return can_add_from_timestep
        if last_max_usage_idx == len(relevant_usages_reversed) - 1:
            return relevant_timesteps[last_max_usage_idx]
        new_timestep = relevant_timesteps[last_max_usage_idx + 1]
        return self.get_timestep_for_tensor_addition(
            tensor, core_id, new_timestep, memory_op
        )

    def generate_all_combinations(self, lst):
        for i in range(1, len(lst) + 1):
            for comb in combinations(lst, i):
                yield comb

    # def find_best_tensor_combination_to_evict(
    #     self,
    #     top_instance,
    #     tensor_to_add,
    #     stored_tensors,
    #     capacity,
    #     tensors_to_avoid_evicting,
    # ):
    #     relevant_tensors_to_avoid_evicting = [
    #         tensor for tensor in tensors_to_avoid_evicting if tensor in stored_tensors
    #     ]
    #     stored_tensors_size = sum(
    #         (stored_tensor.size for stored_tensor in stored_tensors)
    #     )
    #     if stored_tensors_size + tensor_to_add.size <= capacity:
    #         return []
    #     min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
    #     min_score, best_combination_to_evict = float("inf"), []
    #     for combination in self.generate_all_combinations(
    #         [
    #             tensor
    #             for tensor in stored_tensors
    #             if tensor not in relevant_tensors_to_avoid_evicting
    #         ]
    #     ):
    #         score = sum(
    #             (
    #                 stored_tensor.instance_priorities[top_instance] * stored_tensor.size
    #                 for stored_tensor in combination
    #             )
    #         )
    #         evicted_size = sum((stored_tensor.size for stored_tensor in combination))
    #         if evicted_size >= min_size_to_evict and score < min_score:
    #             min_score = score
    #             best_combination_to_evict = list(combination)
    #     if not best_combination_to_evict:
    #         raise ValueError(
    #             "The best tensor combination to evict is empty. tensors_to_avoid_evicting might be too large for the candidate."
    #         )
    #     return best_combination_to_evict

    def find_best_tensor_combination_to_evict_fast(
        self,
        top_instance,
        tensor_to_add,
        timestep,
        exceptions,
    ):
        # Get all tensors that were being stored at the given timestep
        stored_tensors = self.get_tensors_stored_at_timestep(top_instance, timestep)

        # Get the total capacity of this top instance
        capacity = self.top_instance_capacities[top_instance]
        # Sanity check on the tensor we want to add and the memory's capacity
        if capacity < tensor_to_add.size:
            raise ValueError(
                f"Trying to add {tensor_to_add} larger than memory capacity of {top_instance}."
            )

        relevant_exceptions = [
            tensor for tensor in exceptions if tensor in stored_tensors
        ]
        # For the total stored tensors size we also need to take into account all tensors, 
        # including ones that are not yet present at this timestep.
        # Otherwise adding that tensor in the future could cause an overflow.
        stored_tensors_size = self.get_stored_cumsum_at_timestep(top_instance, timestep)
        min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
        if (
            min_size_to_evict <= 0
        ):  # no need to evict any tensor, the memory's space is enough
            return []
        evictable_tensors = [
            tensor for tensor in stored_tensors if tensor not in relevant_exceptions
        ]
        evictable_tensors_priority_size = []
        for tensor in evictable_tensors:
            instance_priority = tensor.get_instance_priority(top_instance, self)
            importance = instance_priority * tensor.size
            evictable_tensors_priority_size.append(importance)

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

    def remove_tensor_from_top_instance(
        self,
        top_instance,
        tensor: Tensor,
        timestep: int,
    ):
        tensor_size = tensor.size
        # Get the instance on the storing core
        try:
            equivalent_tensor = next(
                (
                    stored_tensor
                    for stored_tensor in self.top_instance_stored_tensors[top_instance]
                    if stored_tensor.equality_hash() == tensor.equality_hash()
                )
            )
        except StopIteration:
            # raise ValueError(
            #     f"No tensor found equal to {tensor} in top instance {top_instance}."
            # )
            # If the tensor is not present, we don't have to remove it.
            # This is possible because in "Accelerator.transfer_tensor_to_core(...)"
            # it removes a tensor on a sender core if detects it's no longer needed there.
            return
        self.top_instance_stored_tensors[top_instance].remove(equivalent_tensor)
        del self.top_instance_available_since_timestep[top_instance][
            tensor.equality_hash()
        ]

        self.top_instance_available[top_instance] += tensor_size

        # Use numpy searchsorted to find the where the current_timestep should be inserted
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        insert_idx = np.searchsorted(all_timesteps, timestep)
        timestep_already_present = (
            insert_idx < len(all_timesteps) and all_timesteps[insert_idx] == timestep
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
                [timestep, all_usages[insert_idx - 1] - tensor.size],
                axis=0,
            )

        return

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

    def get_tensors_stored_at_timestep(self, top_instance, timestep):
        stored_tensors = [
            stored_tensor
            for stored_tensor in self.top_instance_stored_tensors[top_instance]
            if self.top_instance_stored_since_timestep[top_instance][
                stored_tensor.equality_hash()
            ]
            <= timestep
        ]
        return stored_tensors

    def get_stored_cumsum_at_timestep(self, top_instance, timestep):
        """
        Return the cumulative size of stored tensors in a top_instance at a timestep.
        """
        stored_cumsum = self.top_instance_stored_cumsum[top_instance]
        timesteps = stored_cumsum[:, 0]
        usages = stored_cumsum[:, 1]
        idx = max(0, np.searchsorted(timesteps, timestep, 'right') - 1)
        return usages[idx]
