import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.memory_instance import MemoryInstance
from zigzag.hardware.architecture.memory_level import MemoryLevel

from stream.hardware.architecture.core import Core
from stream.workload.tensor import Tensor, TensorHash

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages memory allocation, tensor storage, and availability for each core and memory instance.
    Uses hash-based tracking for efficient tensor lookup and removal.
    Handles addition, removal, and querying of tensors in memory.
    """

    def __init__(self, accelerator: "Accelerator") -> None:
        """For each core in the accelerator, create a list containing the top level memories, instances, which memory
        operands they store and their capacity"""

        self.accelerator = accelerator
        self.top_levels: dict[Core, list[MemoryLevel]] = {}
        self.top_instances_per_core: dict[Core, list[MemoryInstance]] = {}

        # memory operand stored by every top level memory
        self.memory_operands_per_core: dict[Core, list[list[MemoryOperand]]] = {}
        for core in self.accelerator.core_list:
            top_levels: list[MemoryLevel] = list(
                (level for level, out_degree in core.memory_hierarchy.out_degree() if out_degree == 0)
            )
            self.top_levels[core] = top_levels
            self.top_instances_per_core[core] = [level.memory_instance for level in top_levels]
            self.memory_operands_per_core[core] = [level.operands for level in top_levels]

        self.unique_top_instances: set[MemoryInstance] = set()
        self.cores_per_top_instance: dict[MemoryInstance, list[Core]] = {}
        self.memory_operands_per_top_instance: dict[MemoryInstance, list[tuple[MemoryOperand, ...]]] = {}

        # Some top level memories instances might be shared, thus we keep info for each unique top memory instance
        self.top_instance_capacities: dict[MemoryInstance, int] = {}
        self.top_instance_available: dict[MemoryInstance, int] = {}
        self.top_instance_stored_tensors: dict[MemoryInstance, list[Tensor]] = {}
        self.top_instance_stored_tensors_hash: dict[MemoryInstance, set[TensorHash]] = {}
        self.top_instance_stored_since_timestep: dict[MemoryInstance, dict[TensorHash, int]] = {}
        self.top_instance_available_since_timestep: dict[MemoryInstance, dict[TensorHash, int]] = {}
        self.top_instance_stored_cumsum: dict[MemoryInstance, np.ndarray[Any, Any]] = {}
        self.top_instance_current_timestep: dict[MemoryInstance, int] = {}
        for core, top_levels in self.top_levels.items():
            for top_level in top_levels:
                top_instance = top_level.memory_instance
                if top_instance not in self.unique_top_instances:
                    self.unique_top_instances.add(top_instance)
                    self.cores_per_top_instance[top_instance] = [core]
                    self.memory_operands_per_top_instance[top_instance] = [tuple(top_level.operands)]
                    self.top_instance_capacities[top_instance] = top_instance.size
                    self.top_instance_available[top_instance] = top_instance.size
                    self.top_instance_stored_tensors[top_instance] = []
                    self.top_instance_stored_tensors_hash[top_instance] = set()
                    self.top_instance_stored_since_timestep[top_instance] = {}
                    self.top_instance_available_since_timestep[top_instance] = {}
                    self.top_instance_stored_cumsum[top_instance] = np.array([[0, 0]])
                    self.top_instance_current_timestep[top_instance] = 0
                else:
                    self.cores_per_top_instance[top_instance].append(core)
                    self.memory_operands_per_top_instance[top_instance].append(tuple(top_level.operands))

        self.offchip_core_id = self.accelerator.offchip_core_id

    def contains(self, tensor: Tensor, top_instance: MemoryInstance):
        """
        Returns True if the tensor (by hash) is present in the given memory instance.
        """
        return tensor.equality_hash in self.top_instance_stored_tensors_hash[top_instance]

    def find_tensor_in_top_instances(self, tensor: Tensor):
        """Find the top memory instances that are storing this tensor."""
        # Find all instances storing this tensor
        instances_storing_tensor: set[MemoryInstance] = set()
        available_since_timesteps: dict[MemoryInstance, int] = {}
        for top_instance, stored_tensor_hashes in self.top_instance_stored_tensors_hash.items():
            if tensor.equality_hash in stored_tensor_hashes:
                instances_storing_tensor.add(top_instance)
                available_since_timesteps[top_instance] = self.top_instance_available_since_timestep[top_instance][
                    tensor.equality_hash
                ]

        if not instances_storing_tensor:
            raise ValueError(f"Tensor {tensor} was not found in any of the instances.")
        return instances_storing_tensor, available_since_timesteps

    def find_tensor(self, tensor: Tensor):
        instances_storing_tensor, available_since_timesteps = self.find_tensor_in_top_instances(tensor)
        cores_storing_tensor: list[int] = []
        top_instance_idxs: list[int] = []
        available_since: list[int] = []
        # Find which cores have these instances as their top instance
        for core, top_instances in self.top_instances_per_core.items():
            for top_instance_idx, top_instance in enumerate(top_instances):
                if top_instance in instances_storing_tensor:
                    cores_storing_tensor.append(core.id)
                    top_instance_idxs.append(top_instance_idx)
                    available_since.append(available_since_timesteps[top_instance])
                    # Remove the entry so that next cores that have this shared memory don't get added
                    instances_storing_tensor.remove(top_instance)
                    del available_since_timesteps[top_instance]

        return cores_storing_tensor, top_instance_idxs, available_since

    def add_tensor(self, tensor: Tensor, core: Core, timestep: int, timestep_end: int, memory_op: MemoryOperand):
        """
        Adds a tensor to the memory of a core at a given timestep, updating all hash-based tracking structures.
        """
        top_level_idx = self.get_top_level_idx(core, memory_op)
        top_instance = self.top_instances_per_core[core][top_level_idx]

        # Check if the tensor is already present
        if self.contains(tensor, top_instance):
            return

        self._add_tensor(tensor, top_instance, timestep_stored=timestep, timestep_available=timestep_end)

        # Use numpy searchsorted to find the where the timestep should be inserted
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        insert_idx = np.searchsorted(all_timesteps, timestep)
        timestep_already_present = insert_idx < len(all_timesteps) and all_timesteps[insert_idx] == timestep

        # We first update the remaining usages of later timesteps
        # If timestep was already in all_timesteps, this timestep will also be updated
        relevant_usages = all_usages[insert_idx:]
        updated_relevant_usages = relevant_usages + tensor.size
        if np.max(updated_relevant_usages, initial=0) > self.top_instance_capacities[top_instance]:
            raise ValueError(f"Inserting {tensor} in {top_instance} caused memory overflow.")
        self.top_instance_stored_cumsum[top_instance][insert_idx:, 1] = updated_relevant_usages

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
        self, tensor: Tensor, core: Core, timestep: int, memory_op: MemoryOperand
    ) -> int:
        """
        Returns the earliest timestep at which the tensor can be added to the core's memory, considering memory usage.

        Args:
            tensor: The tensor to be added to the core.
            core_id: The core id that is going to receive the tensor.
            timestep: The timestep from which to start considering make this tensor data transfer.
            memory_op: The memory operand storing the tensor on the receiving end of the transfer.

        Returns:
            The earliest timestep at which the transfer can actually start.
        """
        top_level_idx = self.get_top_level_idx(core, memory_op)
        top_instance = self.top_instances_per_core[core][top_level_idx]
        top_instance_capacity = self.top_instance_capacities[top_instance]
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        relevant_start_idx = np.searchsorted(all_timesteps, timestep, "right")
        if relevant_start_idx == len(all_timesteps):
            return timestep
        relevant_timesteps = all_timesteps[relevant_start_idx - 1 :]
        relevant_usages = all_usages[relevant_start_idx - 1 :]
        relevant_usages_reversed = relevant_usages[::-1]
        max_usage = np.max(relevant_usages_reversed)
        last_max_usage_idx = len(relevant_usages_reversed) - np.argmax(relevant_usages_reversed) - 1
        if max_usage + tensor.size <= top_instance_capacity:
            can_add_from_timestep = timestep
            return can_add_from_timestep
        if last_max_usage_idx == len(relevant_usages_reversed) - 1:
            return relevant_timesteps[last_max_usage_idx]
        new_timestep = relevant_timesteps[last_max_usage_idx + 1]
        return self.get_timestep_for_tensor_addition(tensor, core, new_timestep, memory_op)

    def find_best_tensor_combination_to_evict_fast(
        self,
        top_instance: MemoryInstance,
        tensor_to_add: Tensor,
        timestep: int,
        exceptions: list[Tensor],
    ) -> list[Tensor]:
        # Get all tensors that were being stored at the given timestep
        stored_tensors = self.get_tensors_stored_at_timestep(top_instance, timestep)

        # Get the total capacity of this top instance
        capacity = self.top_instance_capacities[top_instance]
        # Sanity check on the tensor we want to add and the memory's capacity
        if capacity < tensor_to_add.size:
            raise ValueError(f"Trying to add {tensor_to_add} larger than memory capacity of {top_instance}.")

        relevant_exceptions = [tensor for tensor in exceptions if tensor in stored_tensors]
        # For the total stored tensors size we also need to take into account all tensors,
        # including ones that are not yet present at this timestep.
        # Otherwise adding that tensor in the future could cause an overflow.
        stored_tensors_size = self.get_stored_cumsum_at_timestep(top_instance, timestep)
        min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
        if min_size_to_evict <= 0:  # no need to evict any tensor, the memory's space is enough
            return []
        evictable_tensors = [tensor for tensor in stored_tensors if tensor not in relevant_exceptions]
        evictable_tensors_priority_size: list[int] = []
        for tensor in evictable_tensors:
            instance_priority = tensor.get_instance_priority(top_instance, self)
            importance = instance_priority * tensor.size
            evictable_tensors_priority_size.append(importance)
        evictable_tensors_priority_size_tuple, evictable_tensors_tuple = zip(
            *sorted(zip(evictable_tensors_priority_size, evictable_tensors, strict=False)), strict=False
        )
        evictable_tensors_priority_size = list(evictable_tensors_priority_size_tuple)
        evictable_tensors = list(evictable_tensors_tuple)
        evictable_tensors_size = [tensor.size for tensor in evictable_tensors]
        evictable_tensors_size_sums = [
            sum(evictable_tensors_size[:i]) for i in range(0, len(evictable_tensors_size) + 1)
        ]
        try:
            idx_satisfying_min_size_to_evict = next(
                (i for i, size_sum in enumerate(evictable_tensors_size_sums) if size_sum >= min_size_to_evict)
            )
        except StopIteration as exc:
            raise ValueError(
                f"The evictable tensors {evictable_tensors} and their sizes {evictable_tensors_size} are too small to "
                f"evict a size of {min_size_to_evict}."
            ) from exc
        tensors_to_evict = evictable_tensors[:idx_satisfying_min_size_to_evict]
        return tensors_to_evict

    def remove_tensor_from_top_instance(
        self,
        top_instance: MemoryInstance,
        tensor: Tensor,
        timestep: int,
    ):
        # Get the instance on the storing core
        try:
            equivalent_tensor = next(
                stored_tensor
                for stored_tensor in self.top_instance_stored_tensors[top_instance]
                if stored_tensor.equality_hash == tensor.equality_hash
            )
        except StopIteration:
            # If the tensor is not present, we don't have to remove it. # This is possible because in
            # `Accelerator.transfer_tensor_to_core(...)` it removes a tensor on a sender core if detects it's no longer
            # needed there.
            return

        self.remove_tensor(equivalent_tensor, top_instance)
        # self.top_instance_stored_tensors[top_instance].remove(equivalent_tensor)
        # self.top_instance_stored_tensors_hash[top_instance].remove(tensor.equality_hash)
        # del self.top_instance_available_since_timestep[top_instance][tensor.equality_hash]
        # self.top_instance_available[top_instance] += tensor.size

        # Use numpy searchsorted to find the where the current_timestep should be inserted
        all_timesteps = self.top_instance_stored_cumsum[top_instance][:, 0]
        all_usages = self.top_instance_stored_cumsum[top_instance][:, 1]
        insert_idx = np.searchsorted(all_timesteps, timestep)
        timestep_already_present = insert_idx < len(all_timesteps) and all_timesteps[insert_idx] == timestep

        # We first update the remaining usages of later timesteps
        # If timestep was already in all_timesteps, this timestep will also be updated
        relevant_usages = all_usages[insert_idx:]
        updated_relevant_usages = relevant_usages - tensor.size
        self.top_instance_stored_cumsum[top_instance][insert_idx:, 1] = updated_relevant_usages

        # If the timestep was not in all_timesteps, it will be inserted here
        if not timestep_already_present:
            self.top_instance_stored_cumsum[top_instance] = np.insert(
                self.top_instance_stored_cumsum[top_instance],
                insert_idx,
                [timestep, all_usages[insert_idx - 1] - tensor.size],
                axis=0,
            )

        return

    def get_top_level_idx(self, core: Core, memory_operand: MemoryOperand):
        """Return the index of the top memory that stores memory_operand, index referring to the order in which they
        are stored in the list for this core"""
        return next(
            (
                idx
                for idx, operands_top_level in enumerate(self.memory_operands_per_core[core])
                if memory_operand in operands_top_level
            )
        )

    def get_tensors_stored_at_timestep(self, top_instance: MemoryInstance, timestep: int):
        """
        Returns a list of all tensors stored in the given memory instance at the specified timestep.
        """
        all_stored_tensors = self.top_instance_stored_tensors[top_instance]
        all_stored_tensors_timestep = self.top_instance_stored_since_timestep[top_instance]
        stored_at_timestep = [
            stored_tensor
            for stored_tensor in all_stored_tensors
            if all_stored_tensors_timestep[stored_tensor.equality_hash] <= timestep
        ]
        return stored_at_timestep

    def get_stored_cumsum_at_timestep(self, top_instance: MemoryInstance, timestep: int):
        """
        Return the cumulative size of stored tensors in a top_instance at a timestep.
        """
        stored_cumsum = self.top_instance_stored_cumsum[top_instance]
        timesteps = stored_cumsum[:, 0]
        usages = stored_cumsum[:, 1]
        idx = max(0, np.searchsorted(timesteps, timestep, "right") - 1)  # type: ignore
        return usages[idx]

    def _add_tensor(
        self,
        tensor: Tensor,
        top_instance: MemoryInstance,
        timestep_stored: int,
        timestep_available: int,
    ):
        """
        Internal method to add a tensor to a memory instance, updating all hash-based tracking structures.
        """
        self.top_instance_stored_tensors[top_instance].append(tensor)
        self.top_instance_stored_tensors_hash[top_instance].add(tensor.equality_hash)
        self.top_instance_stored_since_timestep[top_instance][tensor.equality_hash] = timestep_stored
        self.top_instance_available_since_timestep[top_instance][tensor.equality_hash] = timestep_available
        self.top_instance_available[top_instance] -= tensor.size

    def remove_tensor(
        self,
        tensor: Tensor,
        top_instance: MemoryInstance,
    ):
        """
        Removes a tensor from the given memory instance, updating all hash-based tracking structures.
        """
        self.top_instance_stored_tensors[top_instance].remove(tensor)
        self.top_instance_stored_tensors_hash[top_instance].remove(tensor.equality_hash)
        del self.top_instance_available_since_timestep[top_instance][tensor.equality_hash]

        self.top_instance_available[top_instance] += tensor.size
