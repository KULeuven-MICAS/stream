import logging
from collections import defaultdict
from enum import Enum, auto
from math import ceil
from operator import itemgetter
from typing import TYPE_CHECKING

from zigzag.datatypes import Constants, LayerDim, LayerOperand, MemoryOperand

from stream.hardware.architecture.core import Core
from stream.workload.computation.computation_node import ComputationNode, GeneratedComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.tensor import Tensor

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)


class TransferCause(Enum):
    """Enumerates the causes for tensor transfers, used for logging energy in different categories."""

    SINK_LAYER = auto()
    EVICTION = auto()
    OFF_CHIP = auto()
    CORE_TO_CORE = auto()
    NO_LOG = auto()


class CoalaScheduler:
    """
    Schedules computation nodes on an accelerator, handling memory, communication, and subtensor logic.
    Uses a class-based approach for modularity and extensibility.
    Handles splitting of tensors into subtensors when only partial data is needed by a node.
    """

    def __init__(
        self,
        g: ComputationNodeWorkload,
        accelerator: "Accelerator",
        scheduling_order: list[tuple[int, int]],
        cores_idle_from: dict[int, int] | None = None,
        operands_to_prefetch: list[LayerOperand] | None = None,
    ):
        """
        Args:
            G: Graph containing the nodes to be scheduled.
            accelerator: The accelerator to schedule the nodes on.
            scheduling_order: List of (layer_id, sub_id) tuples indicating scheduling order.
            cores_idle_from: Optional dict mapping core_id to start offset.
            operands_to_prefetch: Layer operands to prefetch at the start of the schedule.
        """
        if operands_to_prefetch is None:
            operands_to_prefetch = []
        self.G = g
        self.accelerator = accelerator
        self.scheduling_order = scheduling_order
        self.operands_to_prefetch = operands_to_prefetch

        core_ids = set(n.chosen_core_allocation for n in g.node_list)
        assert None not in core_ids, "Not all nodes have core allocation. Insert SetFixedAllocationPerformanceStage."
        all_core_ids: list[int] = sorted(list(core_ids))  # type: ignore
        self.cores_idle_from = cores_idle_from if cores_idle_from else {core_id: 0 for core_id in all_core_ids}

        # Initialize the schedule results
        self.latency = 0
        self.total_cn_onchip_energy = 0
        self.link_energy: dict[TransferCause, float] = defaultdict(lambda: 0)
        self.memory_energy: dict[TransferCause, float] = defaultdict(lambda: 0)

        # Remains constant throughout the scheduling
        self.sink_layer_nodes = self.get_sink_layer_nodes()
        self.offchip_core = accelerator.get_offchip_core()
        self.offchip_top_instances = self.accelerator.get_top_instances_of_core(self.offchip_core)
        self.nb_graph_nodes = g.number_of_nodes()
        self._initialize_scheduling_order_lookup()

        # Initialize bookkeeping
        self.nb_scheduled_nodes = 0
        self.scheduled_nodes: set[ComputationNode] = set()
        self.bw_fraction_to_use_for_tensor: dict[Tensor, float] = {}
        self.candidates = self.get_initial_candidates()
        self.initialize_tensor_priorities()
        self.initialize_offchip_tensors()

    def _initialize_scheduling_order_lookup(self):
        """
        Initializes lookup dictionaries for fast access from (id, sub_id) to scheduling order index.
        """
        # {item: idx for idx, item in enumerate(self.scheduling_order)}
        self.scheduling_order_lookup: dict[tuple[int, int], int] = {}
        self.scheduling_order_lookup_tiered: dict[int, dict[int, int]] = {}
        for idx, item in enumerate(self.scheduling_order):
            self.scheduling_order_lookup[item] = idx
            layer_id, sub_id = item
            if layer_id not in self.scheduling_order_lookup_tiered:
                self.scheduling_order_lookup_tiered[layer_id] = {sub_id: idx}
            else:
                self.scheduling_order_lookup_tiered[layer_id][sub_id] = idx

    def get_initial_candidates(self):
        """
        Returns the initial candidate nodes (those with no incoming edges) for scheduling.
        """
        candidates: list[tuple[int, ComputationNode]] = []
        for source_node in (n for n, d in self.G.in_degree() if d == 0):
            core_allocation = source_node.chosen_core_allocation
            candidates.append((self.cores_idle_from[core_allocation], source_node))  # type: ignore
        return candidates

    def get_sink_layer_nodes(self):
        """
        Returns all nodes with no successors that produce final outputs (used for off-loading outputs).
        """
        sink_layer_ids = self.G.get_sink_layer_ids()
        sink_layer_nodes = set(n for n in self.G.node_list if (n.id in sink_layer_ids) and n.produces_final_output)
        return sink_layer_nodes

    def initialize_tensor_priorities(self):
        """
        Initializes memory instance priorities for each tensor in the workload.
        """
        for n in self.G.node_list:
            for tensor in n.operand_tensors.values():
                tensor.initialize_instance_priorities(self.G, n, self.accelerator)

    def initialize_offchip_tensors(self):
        """
        Adds constant operand tensors of all nodes to off-chip memory at timestep 0.
        """
        for n in self.G.node_list:
            for op, tensor in n.operand_tensors.items():
                # For constant operands or inputs of first node
                if op in n.constant_operands + n.partially_constant_operands or (
                    op != Constants.OUTPUT_LAYER_OP and len(self.G.in_edges(n)) == 0
                ):
                    if not any(
                        self.accelerator.contains_tensor(tensor, offchip_top_instance)
                        for offchip_top_instance in self.offchip_top_instances
                    ):
                        memory_op = n.memory_operand_links.layer_to_mem_op(op)
                        self.accelerator.spawn(
                            tensor=tensor,
                            core=self.offchip_core,
                            memory_op=memory_op,
                            initial_timestep=0,
                            available_timestep=0,
                        )

    def run(self):
        """
        Main scheduling loop. For each candidate node, handles:
        - Subtensor creation if only a portion of a tensor is needed
        - Memory management and transfer
        - Scheduling and output tensor spawning
        - Memory cleanup and candidate extension
        Returns the total latency of the schedule.
        """
        nb_scheduled_nodes = 0
        done = False
        self.prefetch_constant_operands()
        while not done:
            best_candidate, preds_end = self.pop_best_candidate()
            core = self.get_allocated_core(best_candidate)
            full_tensors_this_candidate_needs, tensors_operands = self.get_tensors_needed_for_node(best_candidate)
            sub_tensors_this_candidate_needs = [
                self.split_tensor_if_needed(t, best_candidate, core, timestep=preds_end)
                for t in full_tensors_this_candidate_needs
            ]
            self.reset_too_large_operands_for_subtensors(
                best_candidate, sub_tensors_this_candidate_needs, tensors_operands, core
            )
            transfer_bw_fraction = self.get_transfer_bandwidth_fraction(best_candidate)

            # Step 0: get the start time: when core is available or predecessors finished
            self.check_and_sync_cores(best_candidate)
            core_idle_from = self.cores_idle_from[core.id]
            timestep = max(core_idle_from, preds_end)

            # Step 1: for operands that are too large to store in the core's memory, clear the memory so ZigZag can
            # optimize the loop ordering using the full memory size
            if best_candidate.too_large_operands:
                transfer_complete_timestep = self.clear_memories(
                    core=core,
                    memory_operands=best_candidate.too_large_operands,
                    timestep=timestep,
                    exceptions=sub_tensors_this_candidate_needs,
                    transfer_bandwidth_fraction=transfer_bw_fraction,
                )
                timestep = transfer_complete_timestep

            # Step 2: Transfer the tensors needed for this node to the core (from off-chip or from another core)
            transfer_headstart = sum(
                ceil(
                    tensor.size
                    / (self.offchip_core.get_top_memory_instance(tensor_operand).ports[0].bw_max * transfer_bw_fraction)
                )
                for tensor, tensor_operand in zip(sub_tensors_this_candidate_needs, tensors_operands, strict=False)
            )
            earliest_t = core_idle_from - transfer_headstart
            for tensor, tensor_operand in zip(sub_tensors_this_candidate_needs, tensors_operands, strict=False):
                transfer_complete_timestep = self.schedule_tensor_transfer(
                    tensor=tensor,
                    tensor_operand=tensor_operand,
                    receiving_core=core,
                    non_evictable_tensors=sub_tensors_this_candidate_needs,
                    earliest_t=earliest_t,
                    transfer_bandwidth_fraction=transfer_bw_fraction,
                )
                timestep = max(timestep, transfer_complete_timestep)

            # Step 3: make space for the output tensor of this node
            output_tensor = best_candidate.get_output_tensor()
            output_memory_operand = output_tensor.memory_operand
            core_to_add_output_to = (
                self.offchip_core if output_memory_operand in best_candidate.too_large_operands else core
            )
            transfer_complete_timestep = self.make_space_for_tensor(
                output_tensor,
                core_to_add_output_to,
                output_memory_operand,
                timestep,
                sub_tensors_this_candidate_needs,
            )
            timestep = transfer_complete_timestep

            # Step 4: If any operands are too large to store in memory, find a window and block off-chip links for the
            # runtime duration
            blocking_can_start_timestep = self.accelerator.block_offchip_links(
                too_large_operands=best_candidate.too_large_operands,
                core_id=core.id,
                start_timestep=timestep,
                duration=best_candidate.get_runtime(),
                cn=best_candidate,
            )
            timestep = blocking_can_start_timestep

            # Step 5: Register the scheduling decision for this node and spawn the output tensor
            node_end_timestep = self.register_scheduled_node(
                node=best_candidate,
                start_time=timestep,
                output_tensor=output_tensor,
                output_memory_operand=output_memory_operand,
                core_to_add_output_to=core_to_add_output_to,
                core_to_run_on=core,
            )
            timestep = node_end_timestep

            # Step 6: manage memory usage when the node ends
            self.decrease_priority(full_tensors_this_candidate_needs, tensors_operands, best_candidate)
            self.check_for_removal(full_tensors_this_candidate_needs, timestep, transfer_bw_fraction)
            self.remove_sub_tensors(
                core,
                sub_tensors_this_candidate_needs,
                tensors_operands,
                timestep=timestep,
                exceptions=full_tensors_this_candidate_needs,
            )
            self.remove_sink_node_tensor(
                node=best_candidate,
                tensor_to_remove=output_tensor,
                core_to_remove_from=core,
                timestep=timestep,
                transfer_bandwidth_fraction=transfer_bw_fraction,
            )

            # Step 7: finish this round
            self.bw_fraction_to_use_for_tensor[output_tensor] = transfer_bw_fraction
            self.extend_candidates(best_candidate)
            nb_scheduled_nodes += 1
            done = nb_scheduled_nodes == self.nb_graph_nodes
        self.latency = self.get_total_latency()
        return self.latency

    def prefetch_constant_operands(self):
        """
        Loads the specified operands_to_prefetch to the cores they belong to at the start of the schedule.
        """
        for n in self.G.node_list:
            for op, tensor in n.operand_tensors.items():
                if op in n.constant_operands and op in self.operands_to_prefetch:
                    core = self.get_allocated_core(n)
                    memory_op = n.memory_operand_links.layer_to_mem_op(op)
                    if not self.accelerator.core_contains_tensor(tensor, core):
                        self.schedule_tensor_transfer(
                            tensor=tensor,
                            tensor_operand=memory_op,
                            receiving_core=core,
                            non_evictable_tensors=[],
                        )

    def pop_best_candidate(self) -> tuple[ComputationNode, int]:
        """
        Returns the best candidate node to schedule next, based on scheduling order priority.
        Removes the candidate from the list.
        """
        if not self.candidates:
            raise ValueError("There are no candidates to schedule.")
        preds_ends, cn_candidates = zip(*self.candidates, strict=False)
        idxs = [self.scheduling_order_lookup[(n.id, n.sub_id)] for n in cn_candidates]
        best_candidate_idx = idxs.index(min(idxs))
        best_candidate = cn_candidates[best_candidate_idx]
        preds_end = preds_ends[best_candidate_idx]
        # Remove the candidate from the list of candidates
        del self.candidates[best_candidate_idx]
        return best_candidate, preds_end

    def split_tensor_if_needed(self, tensor: Tensor, node: ComputationNode, core: Core, timestep: int):
        """
        Returns a subtensor if only a portion of the original tensor is needed by the node.
        If the tensor is already present in the core or no splitting is required, returns the original tensor.
        Handles cases where loop dimensions differ between producer and consumer nodes.
        """
        #  Only do it if the tensor is still in offchip
        if self.accelerator.core_contains_tensor(tensor, core):
            return tensor

        loop_dims_to_split = [dim for dim, _ in node.intra_core_tiling if dim in tensor.loop_dimensions]
        # Generated computation nodes always have a split in their gen split layer dim
        if isinstance(node, GeneratedComputationNode):
            gen_split_dim = node.gen_split_layer_dim
            if gen_split_dim in tensor.loop_dimensions and gen_split_dim not in loop_dims_to_split:
                loop_dims_to_split.append(gen_split_dim)

        if not loop_dims_to_split:
            return tensor

        needed_ranges = [node.loop_ranges[dim] for dim in loop_dims_to_split]
        full_ranges = [tensor.loop_ranges_per_dim[dim] for dim in loop_dims_to_split]

        needed_range_sizes = [r[1] - r[0] for r in needed_ranges]
        full_range_sizes = [r[1] - r[0] for r in full_ranges]

        # No gain in splitting if the tensor is already the right size
        if all(size1 == size2 for size1, size2 in zip(needed_range_sizes, full_range_sizes, strict=False)):
            return tensor

        creation_timestep = self.accelerator.get_available_timestep(tensor, self.offchip_core)

        sub_tensor = self.create_sub_tensor(tensor, loop_dims_to_split, needed_ranges)
        self.accelerator.spawn(
            sub_tensor,
            core=self.offchip_core,
            memory_op=tensor.memory_operand,
            initial_timestep=creation_timestep,
            available_timestep=creation_timestep,
        )
        return sub_tensor

    def create_sub_tensor(
        self, tensor: Tensor, loop_dims_to_split: list[LayerDim], sub_loop_ranges: list[tuple[int, int]]
    ):
        """
        Creates a new Tensor object representing a subtensor with updated loop ranges and size.
        Args:
            tensor: The original tensor to split.
            loop_dims_to_split: List of loop dimensions to split on.
            sub_loop_ranges: The required ranges for each split dimension.
        Returns:
            A new Tensor object representing the subtensor.
        """
        assert all(dim in tensor.loop_dimensions for dim in loop_dims_to_split), (
            f"The loop dimensions to split {loop_dims_to_split} are not in the tensor: {tensor.loop_dimensions}."
        )
        assert loop_dims_to_split, "No loop dimensions to split given."

        new_loop_ranges = list(tensor.loop_ranges)
        compression_factor = 1
        for dim, sub_loop_range in zip(loop_dims_to_split, sub_loop_ranges, strict=False):
            idx = tensor.loop_dimensions.index(dim)
            new_loop_ranges[idx] = sub_loop_range
            full_loop_range = tensor.loop_ranges[idx]
            compression_factor *= (full_loop_range[1] - full_loop_range[0]) / (sub_loop_range[1] - sub_loop_range[0])

        new_loop_ranges = tuple(new_loop_ranges)

        sub_tensor = Tensor(
            size=int(tensor.size / compression_factor),
            origin=tensor.origin,
            layer_operand=tensor.layer_operand,
            loop_dimensions=tensor.loop_dimensions,
            loop_ranges=new_loop_ranges,
        )

        return sub_tensor

    def reset_too_large_operands_for_subtensors(
        self,
        node: ComputationNode,
        sub_tensors_this_candidate_needs: list[Tensor],
        memory_operands: list[MemoryOperand],
        core: Core,
    ):
        """
        Recomputes node.too_large_operands after subtensor splitting.
        If all subtensors now fit in memory, clears too_large_operands for the node.
        """
        if not node.too_large_operands:
            return

        if not sub_tensors_this_candidate_needs:
            return

        # Constant operands that are in `too_large_operands` don't get a tensor, so there can be no smaller sub-tensor
        for layer_op in node.constant_operands + node.partially_constant_operands:
            memory_op = node.memory_operand_links.layer_to_mem_op(layer_op)
            if memory_op in node.too_large_operands:
                return

        # Get total required size for each mem op
        size_per_operand: dict[MemoryOperand, int] = defaultdict(lambda: 0)
        for tensor, memory_operand in zip(sub_tensors_this_candidate_needs, memory_operands, strict=False):
            size_per_operand[memory_operand] += tensor.size
        output_tensor = node.get_output_tensor()
        size_per_operand[output_tensor.memory_operand] += output_tensor.size

        top_memories = set([memory[-1] for (_, memory) in core.mem_hierarchy_dict.items()])
        for mem in top_memories:
            available_capacity = mem.memory_instance.size
            required_capacity = sum(size_per_operand[mem_op] for mem_op in mem.mem_level_of_operands.keys())
            if required_capacity > available_capacity:
                return

        node.set_too_large_operands([])

    def check_and_sync_cores(
        self,
        best_candidate: ComputationNode,
    ):
        """
        Synchronizes cores_idle_from values if the best_candidate is the first node of a layer and sync is needed.
        """
        # Get the predecessor ids of the best_candidate from the workload graph G
        predecessor_ids = (pred.id for pred in self.G.predecessors(best_candidate) if pred.id != best_candidate.id)
        predecessor_idxs: list[int] = []

        for pred_id in predecessor_ids:
            predecessor_idxs += list(self.scheduling_order_lookup_tiered[pred_id].values())

        best_candidate_idx = self.scheduling_order_lookup[(best_candidate.id, best_candidate.sub_id)]

        # Correct way to compute
        # is_first_node_of_layer = not any(
        #     layer_id == best_candidate.id for layer_id, _ in self.scheduling_order[0:best_candidate_idx]
        # )
        # ! Fast way to compute -RG
        is_first_node_of_layer = best_candidate.sub_id == 0 and (
            not isinstance(best_candidate, GeneratedComputationNode) or (best_candidate.gen_id == 0)
        )

        all_predecessors_are_scheduled = all(idx < best_candidate_idx for idx in predecessor_idxs)

        if is_first_node_of_layer and all_predecessors_are_scheduled:
            self._sync_cores()

    def _sync_cores(self):
        """Sync the `cores_idle_from` by setting all timesteps to the latest timestep of all cores. The next layer
        can then start at the same timestep on all cores."""
        max_idle_time = max(self.cores_idle_from.values())
        for core_id in self.cores_idle_from:
            self.cores_idle_from[core_id] = max_idle_time

    def get_tensors_needed_for_node(self, node: ComputationNode):
        """
        Determines all the tensors needed to compute a node, including constant and non-constant operands.
        Returns:
            tensors_this_candidate_needs: list[Tensor]
            tensors_operands: list[MemoryOperand]
        """
        tensors_this_candidate_needs: list[Tensor] = []
        tensors_operands: list[MemoryOperand] = []
        # Constant operands
        for layer_op in node.constant_operands:
            memory_op = node.memory_operand_links.layer_to_mem_op(layer_op)
            if memory_op in node.too_large_operands:
                continue
            tensors_this_candidate_needs.append(node.operand_tensors[layer_op])
            tensors_operands.append(memory_op)
        # Non-constant operands
        for pred, _, edge_data in sorted(self.G.in_edges(node, data=True), key=itemgetter(0)):
            if pred.id == node.id:
                continue  # Skip if predecessor was from the same layer (intra-edge)
            consumer_layer_op = edge_data["operand"]
            consumer_memory_op = node.memory_operand_links[consumer_layer_op]
            if consumer_memory_op in node.too_large_operands:
                continue  # Skip if tensor will be fetched from offchip throughout computation
            pred_output_tensor = pred.operand_tensors[pred.output_operand]
            tensors_this_candidate_needs.append(pred_output_tensor)
            tensors_operands.append(consumer_memory_op)
        if tensors_this_candidate_needs:
            # Sort these tensors based on their earliest possible transfer time
            zipped = list(zip(tensors_this_candidate_needs, tensors_operands, strict=False))
            zipped.sort(key=lambda x: x[0].origin.id if hasattr(x[0], "origin") and hasattr(x[0].origin, "id") else 0)
            tensors_this_candidate_needs, tensors_operands = map(list, zip(*zipped, strict=False))
        return tensors_this_candidate_needs, tensors_operands

    def clear_memories(
        self,
        core: Core,
        memory_operands: list[MemoryOperand],
        timestep: int,
        exceptions: list[Tensor] | None = None,
        transfer_bandwidth_fraction: float = 1,
    ):
        """Remove all tensors from a core's memory for the given  memory operands.
        All tensors are written back to offchip before removal.

        Args:
            core: The Core to remove the tensor from
            memory_operand: The memory operand for which all tensors should be evicted.
            timestep: The timestep to remove the tensor at.
            exceptions: A list of tensors that should not be evicted.
            transfer_bandwidth_fraction: Fraction of the bandwidth to use for the transfers.
        """
        if exceptions is None:
            exceptions = []
        for memory_operand in memory_operands:
            stored_tensors = self.accelerator.get_tensors_stored_in_core(core, memory_operand, timestep)
            for tensor in stored_tensors:
                if tensor not in exceptions:
                    timestep = self.schedule_tensor_removal(
                        tensor_to_remove=tensor,
                        core_to_remove_from=core,
                        memory_op=memory_operand,
                        timestep=timestep,
                        transfer_bandwidth_fraction=transfer_bandwidth_fraction,
                        write_back_to_offchip=True,
                        transfer_cause=TransferCause.EVICTION,
                    )
        return timestep

    def schedule_tensor_removal(
        self,
        tensor_to_remove: Tensor,
        core_to_remove_from: Core,
        memory_op: MemoryOperand,
        timestep: int,
        transfer_bandwidth_fraction: float = 1,
        write_back_to_offchip: bool = False,
        transfer_cause: TransferCause = TransferCause.EVICTION,
    ):
        """Remove tensor from core. If required, transfer to offchip before removal.

        Args:
            tensor: The tensor to remove.
            core: The Core to remove the tensor from.
            memory_op: The memory operand of the tensor.
            timestep: The timestep to remove the tensor at.
            transfer_bandwidth_fraction: Fraction of the bandwidth to use for the transfer.
            write_back_to_offchip: Write the tensor to offchip before removal. Defaults to False.
        """
        should_be_written_to_offchip = write_back_to_offchip and not self.accelerator.core_contains_tensor(
            tensor_to_remove, self.offchip_core
        )
        if should_be_written_to_offchip:
            transfer_end = self.schedule_tensor_transfer(
                tensor=tensor_to_remove,
                receiving_core=self.offchip_core,
                tensor_operand=memory_op,
                sending_core=core_to_remove_from,
                transfer_bandwidth_fraction=transfer_bandwidth_fraction,
                transfer_cause=transfer_cause,
            )

            timestep = max(timestep, transfer_end)

        self.accelerator.remove_tensor(
            tensor=tensor_to_remove, core=core_to_remove_from, memory_op=memory_op, timestep=timestep
        )

        return timestep

    def schedule_tensor_transfer(
        self,
        tensor: Tensor,
        receiving_core: Core,
        tensor_operand: MemoryOperand,
        earliest_t: int = 0,
        non_evictable_tensors: list[Tensor] | None = None,
        sending_core: Core | None = None,
        transfer_bandwidth_fraction: float = 1,
        transfer_cause: TransferCause | None = None,
    ):
        """Find the earliest time to transfer the tensor to the receiving core, and register the transfer.
        Evictions of older tensors might be necessary
        """
        if non_evictable_tensors is None:
            non_evictable_tensors = []
        if self.accelerator.core_contains_tensor(tensor, receiving_core):
            return earliest_t

        tensor_available_since_timestep = self.accelerator.get_available_timestep(tensor, sending_core)
        earliest_tensor_addition_t = max(earliest_t, tensor_available_since_timestep)

        # Evict older tensors if given tensor doesn't fit yet
        evictions_complete_timestep = self.make_space_for_tensor(
            tensor=tensor,
            core=receiving_core,
            memory_op=tensor_operand,
            timestep=earliest_tensor_addition_t,
            tensors_to_avoid_evicting=non_evictable_tensors,
        )

        # Find idle window between sender and receiver cores
        # TODO If the storing_instance is a shared instance across more than one core, there will be multiple possible
        # TODO cores to transfer between. For now, we take the first one
        sending_cores = self.accelerator.get_storing_cores(tensor, sending_core)
        sending_core = sending_cores[0]

        transfer_start, transfer_end = self.accelerator.find_earliest_time_for_transfer(
            tensor=tensor,
            sending_core=sending_core,
            receiving_core=receiving_core,
            earliest_t=evictions_complete_timestep,
            bandwidth_fraction=transfer_bandwidth_fraction,
        )

        # Spawn the tensor on the receiving core, remove from sending core and update communication links
        transfer_link_energy_cost, transfer_memory_energy_cost = self.accelerator.register_tensor_transfer(
            tensor=tensor,
            tensor_operand=tensor_operand,
            sending_core=sending_core,
            receiving_core=receiving_core,
            transfer_start=transfer_start,
            transfer_end=transfer_end,
            transfer_bandwidth_fraction=transfer_bandwidth_fraction,
        )

        # Register energy
        if not transfer_cause:
            came_form_offchip = sending_core == self.offchip_core
            transfer_cause = TransferCause.OFF_CHIP if came_form_offchip else TransferCause.CORE_TO_CORE

        self.link_energy[transfer_cause] += transfer_link_energy_cost
        self.memory_energy[transfer_cause] += transfer_memory_energy_cost

        return transfer_end

    def make_space_for_tensor(
        self,
        tensor: Tensor,
        core: Core,
        memory_op: MemoryOperand,
        timestep: int,
        tensors_to_avoid_evicting: list[Tensor] | None = None,
    ):
        """Make space for the given tensor on the given core by evicting already stored tensors if necessary.

        Args:
            tensor: The tensor to make space for.
            core: The core where the tensor will be stored.
            memory_op: The memory operand on the core.
            timestep: The timestep at which to make space for.
            tensors_to_avoid_evicting: A list of tensors that should not be evicted.
        """
        # Earliest timestep when the core has enough space, or the latest timestep if this is never the case
        enough_space_timestep = self.accelerator.memory_manager.get_timestep_for_tensor_addition(
            tensor=tensor,
            core=core,
            timestep=timestep,
            memory_op=memory_op,
        )

        tensors_to_evict = self.accelerator.find_best_tensor_combination_to_evict_fast(
            tensor=tensor,
            core=core,
            timestep=enough_space_timestep,
            exceptions=tensors_to_avoid_evicting,
        )

        if core == self.offchip_core and tensors_to_evict:
            raise ValueError("Evictions required in offchip memory. Consider making offchip larger.")

        for tensor_to_evict in tensors_to_evict:
            transfer_bandwidth_fraction = self.get_transfer_bandwidth_fraction_for_eviction(tensor_to_evict)
            t_eviction_complete = self.schedule_tensor_removal(
                tensor_to_remove=tensor_to_evict,
                core_to_remove_from=core,
                memory_op=memory_op,
                timestep=timestep,
                transfer_bandwidth_fraction=transfer_bandwidth_fraction,
                write_back_to_offchip=True,
                transfer_cause=TransferCause.EVICTION,
            )
            timestep = max(timestep, t_eviction_complete)

        t_evictions_complete = max(enough_space_timestep, timestep)
        return t_evictions_complete

    def remove_sink_node_tensor(
        self,
        node: ComputationNode,
        tensor_to_remove: Tensor,
        core_to_remove_from: Core,
        timestep: int,
        transfer_bandwidth_fraction: float,
    ):
        """If this node is a sink node (node that has no successors and that produces a final output), transfer final
        outputs to offchip
        """
        if node in self.sink_layer_nodes:
            # Only push back sink node outputs if they're generated and stored on the core
            if Constants.OUTPUT_MEM_OP not in node.too_large_operands:
                self.schedule_tensor_removal(
                    tensor_to_remove=tensor_to_remove,
                    core_to_remove_from=core_to_remove_from,
                    memory_op=tensor_to_remove.memory_operand,
                    timestep=timestep,
                    transfer_bandwidth_fraction=transfer_bandwidth_fraction,
                    write_back_to_offchip=True,
                    transfer_cause=TransferCause.SINK_LAYER,
                )

    def register_scheduled_node(
        self,
        node: ComputationNode,
        start_time: int,
        output_tensor: Tensor,
        output_memory_operand: MemoryOperand,
        core_to_add_output_to: Core,
        core_to_run_on: Core,
    ):
        """Spawn the output tensor and register the runtimes and energies of the node."""

        end_time = start_time + node.get_runtime()
        self.accelerator.spawn(
            output_tensor,
            core_to_add_output_to,
            output_memory_operand,
            initial_timestep=start_time,
            available_timestep=end_time,
        )
        node.set_start(start_time)
        node.set_end(end_time)
        self.cores_idle_from[core_to_run_on.id] = end_time
        self.scheduled_nodes.add(node)

        self.total_cn_onchip_energy += node.get_onchip_energy()
        self.memory_energy[TransferCause.OFF_CHIP] += node.get_offchip_energy()
        return end_time

    def decrease_priority(
        self,
        tensors: list[Tensor],
        tensors_operands: list[MemoryOperand],
        node: ComputationNode,
    ):
        for tensor_used_by_node, tensor_memory_operand in zip(tensors, tensors_operands, strict=False):
            # TODO: tensor_memory_operand will be 'O' for activation tensors.
            # TODO: If the memory between input and output is not shared, this will give a wrong instance.
            assert node.chosen_core_allocation is not None
            top_instance = self.accelerator.get_top_instance_of_core(node.chosen_core_allocation, tensor_memory_operand)
            tensor_used_by_node.instance_priorities[top_instance] -= 1

    def check_for_removal(
        self,
        tensors: list[Tensor],
        timestep: int,
        transfer_bandwidth_fraction: float = 1,
    ):
        """Remove the tensor from the core if its priority is zero."""
        for tensor_used_by_node in tensors:
            if tensor_used_by_node.get_total_priority() == 0:
                instances_storing_tensor, _ = self.accelerator.memory_manager.find_tensor_in_top_instances(
                    tensor_used_by_node
                )
                for instance_storing_tensor in instances_storing_tensor:
                    core_ids_of_instance = [
                        core.id
                        for core in self.accelerator.memory_manager.cores_per_top_instance[instance_storing_tensor]
                    ]
                    # If this tensor is an output tensor, find all nodes that needed it
                    # to get an accurate timestep at which it can be removed
                    timestep_for_removal = timestep
                    if tensor_used_by_node.layer_operand == tensor_used_by_node.origin.output_operand:
                        origin = tensor_used_by_node.origin
                        if self.offchip_core.id in core_ids_of_instance:
                            # If wanting to discard it from offchip, look at the max end time across all successors
                            nodes_that_needed_tensor = [n for n in self.G.successors(origin) if n.id != origin.id]
                        else:
                            # If discarding it from a regular core, look at the max end time successors that used it
                            # from that instance
                            nodes_that_needed_tensor = [
                                n
                                for n in self.G.successors(origin)
                                if n.chosen_core_allocation in core_ids_of_instance and n.id != origin.id
                            ]
                        end_times = [n.end for n in nodes_that_needed_tensor if n.end >= 0]
                        max_end_time = max(end_times, default=timestep_for_removal)
                        # assert max_end_time != -1, "There should be at least one successor."
                        timestep_for_removal = max_end_time

                    # Get a core tied to the top_instance we want to remove it on.
                    core = self.accelerator.memory_manager.cores_per_top_instance[instance_storing_tensor][0]
                    self.schedule_tensor_removal(
                        tensor_to_remove=tensor_used_by_node,
                        core_to_remove_from=core,
                        memory_op=tensor_used_by_node.memory_operand,
                        timestep=timestep_for_removal,
                        transfer_bandwidth_fraction=transfer_bandwidth_fraction,
                        transfer_cause=TransferCause.NO_LOG,
                    )

    def remove_sub_tensors(
        self,
        core: Core,
        sub_tensors: list[Tensor],
        tensors_operands: list[MemoryOperand],
        exceptions: list[Tensor],
        timestep: int,
    ):
        """Remove all the sub-tensors from the given core, except for the tensors in the exceptions list."""
        for sub_tensor, tensor_operand in zip(sub_tensors, tensors_operands, strict=False):
            if sub_tensor not in exceptions:
                self.accelerator.remove_tensor(sub_tensor, core, tensor_operand, timestep)
                self.accelerator.remove_tensor(sub_tensor, self.offchip_core, tensor_operand, timestep)

    def extend_candidates(self, node: ComputationNode):
        """For each successor of this node, check if all of its predecessors have been scheduled"""
        for successor in sorted(self.G.successors(node)):
            if all(pred in self.scheduled_nodes for pred in self.G.predecessors(successor)):
                preds_end = max(
                    (predecessor.end for predecessor in self.G.predecessors(successor)),
                    default=0,
                )
                self.candidates.append((preds_end, successor))

    def get_total_latency(self):
        """The total schedule latency is the max of all CN end times and the link end times"""
        cns_end_time = max(n.end for n in self.G.node_list)
        links_end_time = max([event.end for event in self.accelerator.communication_manager.events], default=0)
        return max(cns_end_time, links_end_time)

    def get_allocated_core(self, node: ComputationNode):
        """Get the core this candidate will be scheduled on"""
        core_id = node.chosen_core_allocation
        assert core_id is not None
        return self.accelerator.get_core(core_id)

    def get_transfer_bandwidth_fraction(self, node: ComputationNode):
        """Get the fraction of the off-chip bandwidth to be used for the tensor transfers related to this node.
        The fraction should be inversely proportional to how many nodes are expected to run in parallel.
        We assume this is the number of inter-core splits.

        NOTE this assumes all inter-core split nodes can run parallel, but this is not the case if the node has too
        large operands. In this case, we could assume the number of parallel nodes  as the number of nodes that can
        block with the current required blocking bandwidth. However, this does not incorporate broadcasting mechanism
        and is too pessimistic. We do not use this for now as it deteriorates the schedule.
        > if node.too_large_operands:
        >     required_bw = sum(
        >         CommunicationManager.get_instantaneous_offchip_bandwidth(node, op) for op in node.too_large_operands
        >     )
        >     # Assume all mem ops use the same instance and r_bw == w_bw == rw_bw
        >     available_offchip_bw = max(mem.r_bw for mem in self.offchip_top_instances)
        >     possible_parallel_nb_nodes = max(1, available_offchip_bw // required_bw)
        """
        possible_parallel_nb_nodes = node.get_total_inter_core_splits()
        return 1 / possible_parallel_nb_nodes

    def get_transfer_bandwidth_fraction_for_eviction(self, tensor: Tensor):
        """Get the fraction of the off-chip bandwidth to be used to evict this tensor at the given timestep.
        Instead of using the total inter-core splits of the current node, we use the inter-core tiling (i.e. the number
        of cores dealing with the tensor) of the source node of this tensor.
        """
        nb_cores_storing_similar_tensor = tensor.origin.get_total_inter_core_splits()
        return 1 / nb_cores_storing_similar_tensor

    @property
    def total_cn_offchip_link_energy(self):
        return self.link_energy[TransferCause.OFF_CHIP]

    @property
    def total_cn_offchip_memory_energy(self):
        return self.memory_energy[TransferCause.OFF_CHIP]

    @property
    def total_eviction_to_offchip_link_energy(self):
        return self.link_energy[TransferCause.EVICTION]

    @property
    def total_eviction_to_offchip_memory_energy(self):
        return self.memory_energy[TransferCause.EVICTION]

    @property
    def total_sink_layer_output_offchip_link_energy(self):
        return self.link_energy[TransferCause.SINK_LAYER]

    @property
    def total_sink_layer_output_offchip_memory_energy(self):
        return self.memory_energy[TransferCause.SINK_LAYER]

    @property
    def total_core_to_core_link_energy(self):
        return self.link_energy[TransferCause.CORE_TO_CORE]

    @property
    def total_core_to_core_memory_energy(self):
        return self.memory_energy[TransferCause.CORE_TO_CORE]
