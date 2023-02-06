from itertools import combinations

# from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.tensor import Tensor
import bisect
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Class that keeps track of the memory state of all top level memories of each core.
    """

    def __init__(self, accelerator) -> None:
        self.accelerator = accelerator
        # For each core in the accelerator, create a list containing the top level memories, which memory operands they store and their capacity
        self.top_levels = {}  # top level memory of each core
        self.memory_operands = {}  # memory operand stored by every top level memory
        self.capacities = {}  # memory capacity of each top level memory
        self.available = {}  # available memory capacity in bits of each top level memory
        self.delta_history = {}  # tracks used memory space in bits deltas of each top level memory
        self.stored_tensors = {}  # track which tensors are present in each top level memory through a dict {equality_hash: tensor}
        self.stored_since_timestep = {}  # tracks for each tensor since what timestep is has been present in the memory (used for causality)
        self.stored_cumsum = {}
        self.current_timestep = {}
        self.cores = []
        for core in self.accelerator.cores.nodes():
            self.cores.append(core)
            top_levels = list((level for level, out_degree in core.memory_hierarchy.out_degree() if out_degree == 0))
            self.top_levels[core] = top_levels
            self.memory_operands[core] = [level.operands for level in top_levels]
            self.capacities[core] = [level.memory_instance.size for level in top_levels]
            self.available[core] = [level.memory_instance.size for level in top_levels]
            self.stored_tensors[core] = [list() for level in top_levels]
            self.stored_since_timestep[core] = [{} for level in top_levels]
            self.delta_history[core] = [[[0, 0]] for level in top_levels]  # (timestep, delta in used [+ means more used so less available])
            self.stored_cumsum[core] = [[[0, 0]] for level in top_levels]
            self.current_timestep[core] = [0 for level in top_levels]
        self.off_chip_core_id = self.accelerator.offchip_core_id

    def contains(self, tensor: Tensor, core_id: int):
        core = self.accelerator.get_core(core_id)
        memory_op = tensor.memory_operand
        top_level_idx = self.get_top_level_idx(core, memory_op)
        return any([tensor.equality_hash() == stored_tensor.equality_hash() for stored_tensor in self.stored_tensors[core][top_level_idx]])

    def find_tensor(self, tensor: Tensor):
        equality_hash = tensor.equality_hash()
        cores_storing_tensor = []
        top_level_idxs = []
        stored_since = []
        for core in self.stored_tensors:
            for top_level_idx, stored_tensors in enumerate(self.stored_tensors[core]):
                if any([equality_hash == stored_tensor.equality_hash() for stored_tensor in stored_tensors]):
                    cores_storing_tensor.append(core.id)
                    stored_since.append(self.stored_since_timestep[core][top_level_idx][equality_hash])
                    top_level_idxs.append(top_level_idx)
        if not cores_storing_tensor:
            raise ValueError(f"Tensor {tensor} was not found in any of the cores.")
        return cores_storing_tensor, top_level_idxs, stored_since

    def add_tensor_to_core(self, tensor: Tensor, core_id: int, timestep: int, timestep_end: int, tensors_to_avoid_evicting: list, memory_op: str = None):
        timestep_delta = timestep_end - timestep
        total_eviction_link_energy_cost = 0
        total_eviction_memory_energy_cost = 0
        core = self.accelerator.get_core(core_id)
        tensor_size = tensor.size
        if not memory_op:
            memory_op = tensor.memory_operand
        top_level_idx = self.get_top_level_idx(core, memory_op)
        stored_tensors = self.stored_tensors[core][top_level_idx]
        # current_timestep = self.current_timestep[core][top_level_idx]
        # if timestep <= current_timestep:
        #     timestep = current_timestep
        # else:
        #     self.current_timestep[core][top_level_idx] = timestep
        if self.contains(tensor, core_id):
            return
        # If there is no equivalent tensor in the core, remove tensors until we have enough space
        # Tensors are removed based on their priority value
        memory_capacity = self.capacities[core][top_level_idx]
        tensors_to_evict = self.find_best_tensor_combination_to_evict_fast(core_id, tensor, stored_tensors, memory_capacity, tensors_to_avoid_evicting)
        for tensor_to_evict in tensors_to_evict:
            end_of_eviction_timestep, eviction_link_energy_cost, eviction_memory_energy_cost = \
                self.remove_tensor_from_core(core, top_level_idx, tensor_to_evict, timestep, write_back_to_offchip=True)
            if end_of_eviction_timestep > timestep:
                timestep = end_of_eviction_timestep
            total_eviction_link_energy_cost += eviction_link_energy_cost
            total_eviction_memory_energy_cost += eviction_memory_energy_cost

        # Now that we have enough space, we add this tensor
        self.stored_tensors[core][top_level_idx].append(tensor)
        self.stored_since_timestep[core][top_level_idx][tensor.equality_hash()] = timestep + timestep_delta
        self.available[core][top_level_idx] -= tensor_size
        # use package bisect to insert the [timestep, tensor_size] in the correct timeframe, enable data preloading
        bisect.insort(self.delta_history[core][top_level_idx], [timestep, tensor_size])

        (last_timestep, last_cumsum) = self.stored_cumsum[core][top_level_idx][-1]
        if timestep == last_timestep:
            self.stored_cumsum[core][top_level_idx][-1] = [timestep, last_cumsum + tensor_size]
        # if the timestep is before the last_timestep, it means data loading happens, and all the stored_cumsum afterwards need to be updated
        elif timestep < last_timestep:
            insert_id = bisect.bisect(self.stored_cumsum[core][top_level_idx], [timestep, tensor_size])
            already_stored_size = self.stored_cumsum[core][top_level_idx][insert_id - 1][1]
            bisect.insort(self.stored_cumsum[core][top_level_idx], [timestep, already_stored_size + tensor_size])
            for stored_cumsum_afterwards in self.stored_cumsum[core][top_level_idx][insert_id + 1:]:
                stored_cumsum_afterwards[1] += tensor_size
        else:
            self.stored_cumsum[core][top_level_idx].append([timestep, last_cumsum + tensor_size])

        return timestep, total_eviction_link_energy_cost, total_eviction_memory_energy_cost

    def test_add_tensor_to_core(self, tensor: Tensor, core_id: int, test_timestep: int, worst_case_timestep: int, memory_op: str) -> int:
        """
        This function gives the earliest timestep since test_timestep that the tensor can be added to the core

        Args:
        tensor (Tensor): The tensor to be added to the core.
        core_id (int): The core id that is going to receive the tensor.
        test_timestep (int): The timestep from which to start considering make this tensor data transfer.
        memory_op (str): The memory operand storing the tensor on the receiving end of the transfer.
        worst_case_timestep (int): when the data cannot be prefetched (no enough space), the latest timestep that it needs to be transferred.

        Returns:
        can_transfer_from_timestep (int): The earliest timestep at which the transfer can actually start.
        """

        core = self.accelerator.get_core(core_id)
        top_level_idx = self.get_top_level_idx(core, memory_op)
        memory_usage_in_receiver_core = self.stored_cumsum[core][top_level_idx]
        memory_usage_in_receiver_core_when_data_is_ready = None
        if len(memory_usage_in_receiver_core) == 1:
            memory_usage_in_receiver_core_when_data_is_ready = memory_usage_in_receiver_core
        else:
            for idx, memory_usage in enumerate(memory_usage_in_receiver_core):
                if memory_usage[0] > test_timestep:
                    memory_usage_in_receiver_core_when_data_is_ready = memory_usage_in_receiver_core[idx - 1:]
                    break
            if not memory_usage_in_receiver_core_when_data_is_ready:
                memory_usage_in_receiver_core_when_data_is_ready = [memory_usage_in_receiver_core[-1]]

        for memory_usage in memory_usage_in_receiver_core_when_data_is_ready:
            if tensor.size < self.capacities[core][top_level_idx] - memory_usage[1]:
                can_transfer_from_timestep = max(memory_usage[0], test_timestep)
                # if can_transfer_from_timestep >= worst_case_timestep:
                #     logger.warning(f"{tensor} cannot be prefetched to core {core_id}. Cause stall.")
                # else:
                #     logger.info(f"{tensor} is prefetched to core {core_id} to hide stall.")
                return can_transfer_from_timestep
        # logger.warning(f"Tensor {tensor} cannot be prefetched to core {core_id}. Cause stall.")
        return worst_case_timestep

    def generate_all_combinations(self, lst):
        for i in range(1, len(lst) + 1):
            for comb in combinations(lst, i):
                yield comb

    def find_best_tensor_combination_to_evict(self, core_id, tensor_to_add, stored_tensors, capacity, tensors_to_avoid_evicting):
        relevant_tensors_to_avoid_evicting = [tensor for tensor in tensors_to_avoid_evicting if tensor in stored_tensors]
        stored_tensors_size = sum((stored_tensor.size for stored_tensor in stored_tensors))
        if stored_tensors_size + tensor_to_add.size <= capacity:
            return []
        min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
        min_score, best_combination_to_evict = float('inf'), []
        for combination in self.generate_all_combinations([tensor for tensor in stored_tensors if tensor not in relevant_tensors_to_avoid_evicting]):
            score = sum((stored_tensor.core_priorities[core_id] * stored_tensor.size for stored_tensor in combination))
            evicted_size = sum((stored_tensor.size for stored_tensor in combination))
            if evicted_size >= min_size_to_evict and score < min_score:
                min_score = score
                best_combination_to_evict = list(combination)
        if not best_combination_to_evict:
            raise ValueError("The best tensor combination to evict is empty. tensors_to_avoid_evicting might be too large for the candidate.")
        return best_combination_to_evict

    def find_best_tensor_combination_to_evict_fast(self, core_id, tensor_to_add, stored_tensors, capacity, tensors_to_avoid_evicting):
        relevant_tensors_to_avoid_evicting = [tensor for tensor in tensors_to_avoid_evicting if tensor in stored_tensors]
        stored_tensors_size = sum((stored_tensor.size for stored_tensor in stored_tensors))
        min_size_to_evict = tensor_to_add.size - (capacity - stored_tensors_size)
        if min_size_to_evict < 0:   # no need to evict any tensor, the memory's space is enough
            return []
        evictable_tensors = [tensor for tensor in stored_tensors if tensor not in relevant_tensors_to_avoid_evicting]
        evictable_tensors_priority_size = [tensor.core_priorities[core_id] * tensor.size for tensor in evictable_tensors]
        if not evictable_tensors:
            evictable_tensors_priority_size, evictable_tensors = [], []
        else:
            evictable_tensors_priority_size, evictable_tensors = zip(*sorted(zip(evictable_tensors_priority_size, evictable_tensors)))  # sort them
        evictable_tensors_size = [tensor.size for tensor in evictable_tensors]
        evictable_tensors_size_sums = [sum(evictable_tensors_size[:i]) for i in range(0, len(evictable_tensors_size) + 1)]
        try:
            idx_satisfying_min_size_to_evict = next((i for i, size_sum in enumerate(evictable_tensors_size_sums) if size_sum >= min_size_to_evict))
        except StopIteration:
            raise ValueError(f"The evictable tensors {evictable_tensors} and their sizes {evictable_tensors_size} are too small to evict a size of {min_size_to_evict}.")
        tensors_to_evict = evictable_tensors[:idx_satisfying_min_size_to_evict]
        return tensors_to_evict

    def remove_tensor_from_core(self, core, top_level_idx, tensor: Tensor, timestep: int, write_back_to_offchip: bool = True):
        tensor_size = tensor.size

        # Transfer the tensor to off-chip if it's not present there
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        should_be_written_to_offchip = write_back_to_offchip and not self.contains(tensor, self.off_chip_core_id)
        # current_timestep = max(timestep, self.current_timestep[core][top_level_idx])
        current_timestep = timestep
        if should_be_written_to_offchip:
            transfer_start, transfer_end, link_energy_cost, memory_energy_cost = self.accelerator.transfer_data(tensor, core, self.off_chip_core_id, tensor.memory_operand, current_timestep)
            current_timestep = transfer_end
            total_link_energy_cost += link_energy_cost
            total_memory_energy_cost += memory_energy_cost
            self.add_tensor_to_core(tensor, self.off_chip_core_id, transfer_start, transfer_end, [])  # no tensors to avoid evicting on offchip core
            # self.current_timestep[core][top_level_idx] = current_timestep

        try:
            equivalent_tensor = next((stored_tensor for stored_tensor in self.stored_tensors[core][top_level_idx] if stored_tensor.equality_hash() == tensor.equality_hash()))
        except StopIteration:
            raise ValueError(f"No tensor found equal to {tensor} in core {core} top_level_idx {top_level_idx}.")
        self.stored_tensors[core][top_level_idx].remove(equivalent_tensor)
        del self.stored_since_timestep[core][top_level_idx][tensor.equality_hash()]

        self.available[core][top_level_idx] += tensor_size
        bisect.insort(self.delta_history[core][top_level_idx], [current_timestep, -tensor_size])

        (last_timestep, last_cumsum) = self.stored_cumsum[core][top_level_idx][-1]
        if current_timestep == last_timestep:
            self.stored_cumsum[core][top_level_idx][-1] = [current_timestep, last_cumsum - tensor_size]
        elif timestep < last_timestep:
            insert_id = bisect.bisect(self.stored_cumsum[core][top_level_idx], [timestep, -tensor_size])
            already_stored_size = self.stored_cumsum[core][top_level_idx][insert_id][1]
            bisect.insort(self.stored_cumsum[core][top_level_idx], [timestep, already_stored_size - tensor_size])
            for stored_cumsum_afterwards in self.stored_cumsum[core][insert_id + 1:]:
                stored_cumsum_afterwards[1] -= tensor_size
        else:
            self.stored_cumsum[core][top_level_idx].append([current_timestep, last_cumsum - tensor_size])

        return current_timestep, total_link_energy_cost, total_memory_energy_cost

    def evict_all_tensors_from_core(self, core_id, memory_operand, timestep, tensors_to_avoid_evicting):
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
        # stored_tensors = self.stored_tensors[core][top_level_idx]
        for tensor in self.stored_tensors[core][top_level_idx].copy():
            if tensor in tensors_to_avoid_evicting:
                continue  # Skip evicting this tensor (e.g. it's an input for this candidate)
            timestep, link_energy_cost, memory_energy_cost = self.remove_tensor_from_core(core, top_level_idx, tensor, timestep, write_back_to_offchip=True)
            total_link_energy_cost += link_energy_cost
            total_memory_energy_cost += memory_energy_cost
        return total_link_energy_cost, total_memory_energy_cost, timestep

    def get_top_level_idx(self, core, memory_operand):
        """Return the index of the top memory that stores memory_operand, index referring to the order in which they are stored in the list for this core

        Args:
            core (_type_): _description_
            memory_operand (_type_): _description_
        """
        return next((idx for idx, operands_top_level in enumerate(self.memory_operands[core]) if memory_operand in operands_top_level))
