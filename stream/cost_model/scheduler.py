import logging
from collections import defaultdict
from enum import Enum, auto
from operator import itemgetter
from typing import TYPE_CHECKING

from zigzag.datatypes import Constants, LayerOperand, MemoryOperand

from stream.hardware.architecture.core import Core
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.tensor import Tensor

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)


class TransferCause(Enum):
    """Log transfer energies in different categories"""

    SINK_LAYER = auto()
    EVICTION = auto()
    OFF_CHIP = auto()
    CORE_TO_CORE = auto()
    NO_LOG = auto()


class Schedule:
    def __init__(
        self,
        G: ComputationNodeWorkload,
        accelerator: "Accelerator",
        scheduling_order: list[tuple[int, int]],
        cores_idle_from: dict[int, int] | None = None,
        operands_to_prefetch: list[LayerOperand] = [],
    ):
        """
        Args:
            G: Graph containing the nodes to be scheduled.
            accelerator: The accelerator to schedule the nodes on.
            scheduling_order:
            cores_idle_from: A dict containing for each core_id its start offset. Defaults to None.
            operands_to_prefetch: The layer operands that should be prefetched at the start of the schedule.
        """
        self.G = G
        self.accelerator = accelerator
        self.scheduling_order = scheduling_order
        self.operands_to_prefetch = operands_to_prefetch

        core_ids = set(n.chosen_core_allocation for n in G.node_list)
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
        self.nb_graph_nodes = G.number_of_nodes()

        # Initialize bookkeeping
        self.nb_scheduled_nodes = 0
        self.scheduled_nodes: set[ComputationNode] = set()
        self.bw_fraction_to_use_for_tensor: dict[Tensor, float] = {}
        self.candidates = self.get_initial_candidates()
        self.initialize_tensor_priorities()
        self.initialize_offchip_tensors()

    def get_initial_candidates(self):
        """Put the very first nodes of a layer that doesn't have any incoming edges as the first candidates"""
        candidates: list[tuple[int, ComputationNode]] = []
        for source_node in (n for n, d in self.G.in_degree() if d == 0):
            core_allocation = source_node.chosen_core_allocation
            candidates.append((self.cores_idle_from[core_allocation], source_node))  # type: ignore
        return candidates

    def get_sink_layer_nodes(self):
        """Get all the nodes with no successors that produce final outputs, used for off-loading final outputs"""
        sink_layer_ids = self.G.get_sink_layer_ids()
        sink_layer_nodes = set((n for n in self.G.node_list if (n.id in sink_layer_ids) and n.produces_final_output))
        return sink_layer_nodes

    def initialize_tensor_priorities(self):
        """Initialize the memory instance priorities for each tensor in the workload."""
        for n in self.G.node_list:
            for tensor in n.operand_tensors.values():
                tensor.initialize_instance_priorities(self.G, n, self.accelerator)

    def initialize_offchip_tensors(self):
        """Add the constant operand tensors of all nodes to the off-chip initially."""
        offchip_top_instances = self.accelerator.get_top_instances_of_core(self.offchip_core)
        for n in self.G.node_list:
            for op, tensor in n.operand_tensors.items():
                # For constant operands or inputs of first node
                if op in n.constant_operands or (op != Constants.OUTPUT_LAYER_OP and len(self.G.in_edges(n)) == 0):
                    if not any(
                        (
                            self.accelerator.contains_tensor(tensor, offchip_top_instance)
                            for offchip_top_instance in offchip_top_instances
                        )
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
        nb_scheduled_nodes = 0
        done = False

        self.prefetch_constant_operands()

        while not done:
            best_candidate, preds_end = self.pop_best_candidate()
            tensors_this_candidate_needs, tensors_operands = self.get_tensors_needed_for_node(best_candidate)
            core = self.get_allocated_core(best_candidate)
            transfer_bw_fraction = self.get_transfer_bandwidth_fraction(best_candidate)

            # Step 0: get the start time: when core is available or predecessors finished
            self.sync_cores_idle_from(best_candidate)
            core_idle_from = self.cores_idle_from[core.id]
            timestep = max(core_idle_from, preds_end)

            # Step 1: for operands that are too large to store in the core's memory, clear the memory so ZigZag can
            # optimize the loop ordering using the full memory size
            if best_candidate.too_large_operands:
                transfer_complete_timestep = self.clear_memories(
                    core=core,
                    memory_operands=best_candidate.too_large_operands,
                    timestep=timestep,
                    exceptions=tensors_this_candidate_needs,
                    transfer_bandwidth_fraction=transfer_bw_fraction,
                )
                timestep = transfer_complete_timestep

            # Step 2: Transfer the tensors needed for this node to the core (from off-chip or from another core)
            for tensor, tensor_operand in zip(tensors_this_candidate_needs, tensors_operands):
                transfer_complete_timestep = self.schedule_tensor_transfer(
                    tensor=tensor,
                    tensor_operand=tensor_operand,
                    receiving_core=core,
                    non_evictable_tensors=tensors_this_candidate_needs,
                    earliest_t=core_idle_from,
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
                tensors_this_candidate_needs,
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
            self.decrease_priority(tensors_this_candidate_needs, tensors_operands, best_candidate)
            self.check_for_removal(tensors_this_candidate_needs, timestep, transfer_bw_fraction)
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
        """Load the `operands_to_prefetch` to the cores they belong to."""
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
        """Get the best candidate node to schedule next, given the selection priority. Remove that candidate from the
        list of candidates and return it."""
        if not self.candidates:
            raise ValueError("There are no candidates to schedule.")
        preds_ends, cn_candidates = zip(*self.candidates)
        cn_candidates: list[ComputationNode]
        idxs = [self.scheduling_order.index((n.id, n.sub_id)) for n in cn_candidates]
        best_candidate_idx = idxs.index(min(idxs))
        best_candidate = cn_candidates[best_candidate_idx]
        preds_end = preds_ends[best_candidate_idx]
        # Remove the candidate from the list of candidates
        del self.candidates[best_candidate_idx]
        return best_candidate, preds_end

    def sync_cores_idle_from(
        self,
        best_candidate: ComputationNode,
    ):
        """
        Sync the cores_idle_from dict values if the best candidate is the first node of a layer and we detect
        layer-by-layer execution. The layer-by-layer execution is detected through the scheduling_order.
        """
        # Get the predecessor ids of the best_candidate from the workload graph G
        predecessor_ids = [pred.id for pred in self.G.predecessors(best_candidate) if pred.id != best_candidate.id]
        predecessor_idxs = [
            i for i in range(len(self.scheduling_order)) if self.scheduling_order[i][0] in predecessor_ids
        ]

        best_candidate_idx = self.scheduling_order.index((best_candidate.id, best_candidate.sub_id))
        if self.scheduling_order[best_candidate_idx - 1][0] != best_candidate.id and all(
            (i < best_candidate_idx for i in predecessor_idxs)
        ):
            # If the best_candidate is the first node of a layer and all nodes of predecessor layers have been scheduled
            # Sync the cores_idle_from dict
            max_idle_time = max(self.cores_idle_from.values())
            for core_id in self.cores_idle_from:
                self.cores_idle_from[core_id] = max_idle_time

    def get_tensors_needed_for_node(self, node: ComputationNode):
        """Determine all the tensors needed to compute a node.
        The node might need multiple outputs from previous nodes, depending on the graph.

        Args:
            node (ComputationNode): The node to be computed.
            G : The graph of all nodes.

        Returns:
            A tuple of tensors and a tuple of memory operands for the node.
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
        for pred, node, edge_data in sorted(self.G.in_edges(node, data=True), key=itemgetter(0)):
            if pred.id == node.id:
                continue  # Skip if predecessor was from the same layer (intra-edge)
            consumer_layer_op: LayerOperand = edge_data["operand"]
            consumer_memory_op = node.memory_operand_links.layer_to_mem_op(consumer_layer_op)
            if consumer_memory_op in node.too_large_operands:
                continue  # Skip if tensor will be fetched fromm offchip throughout computation
            pred_output_tensor = pred.operand_tensors[pred.output_operand]
            tensors_this_candidate_needs.append(pred_output_tensor)
            tensors_operands.append(consumer_memory_op)
        if tensors_this_candidate_needs:
            # Sort these tensors based on their earliest possible transfer time
            tensors_this_candidate_needs, tensors_operands = zip(
                *sorted(zip(tensors_this_candidate_needs, tensors_operands))
            )
        return tensors_this_candidate_needs, tensors_operands

    def clear_memories(
        self,
        core: Core,
        memory_operands: list[MemoryOperand],
        timestep: int,
        exceptions: list[Tensor] = [],
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
        non_evictable_tensors: list[Tensor] = [],
        sending_core: Core | None = None,
        transfer_bandwidth_fraction: float = 1,
        transfer_cause: TransferCause | None = None,
    ):
        """Find the earliest time to transfer the tensor to the receiving core, and register the transfer.
        Evictions of older tensors might be necessary
        """

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
        tensors_to_avoid_evicting: list[Tensor] = [],
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
            transfer_bandwidth_fraction = self.get_transfer_bandwidth_fraction_for_eviction(tensor_to_evict, timestep)
            t_eviction_complete = self.schedule_tensor_removal(
                tensor_to_remove=tensor_to_evict,
                core_to_remove_from=core,
                memory_op=memory_op,
                timestep=timestep,  # TODO should this be `enough_space_timestep`?
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
        # TODO hier wordt denk ik gene timestep doorgegeven!

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
        for tensor_used_by_node, tensor_memory_operand in zip(tensors, tensors_operands):
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
                            # If discarding it from a regular core, look at the max end time successors that used it from
                            # that instance
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

    def extend_candidates(self, node: ComputationNode):
        """For each successor of this node, check if all of its predecessors have been scheduled"""
        for successor in sorted(self.G.successors(node)):
            if all((pred in self.scheduled_nodes for pred in self.G.predecessors(successor))):
                preds_end = max(
                    (predecessor.end for predecessor in self.G.predecessors(successor)),
                    default=0,
                )
                self.candidates.append((preds_end, successor))

    def get_total_latency(self):
        """The total schedule latency is the max of all CN end times and the link end times"""
        cns_end_time = max((n.end for n in self.G.node_list))
        links_end_time = max([event.end for event in self.accelerator.communication_manager.events], default=0)
        return max(cns_end_time, links_end_time)

    def get_allocated_core(self, node: ComputationNode):
        """Get the core this candidate will be scheduled on"""
        core_id = node.chosen_core_allocation
        assert core_id is not None
        return self.accelerator.get_core(core_id)

    def get_transfer_bandwidth_fraction(self, node: ComputationNode):
        """Get the fraction of the off-chip bandwidth to be used for the tensor transfers related to this node"""
        return 1 / node.get_total_inter_core_splits()

    def get_transfer_bandwidth_fraction_for_eviction(self, tensor: Tensor, timestep: int):
        """Get the fraction of the off-chip bandwidth to be used to evict this tensor at the given timestep.
        Instead of using the total inter-core splits of the current node, we use the number of cores that store a tensor
        of the same layer and memory operand at the given timestep.
        # TODO check for given timestep
        """

        def contains_related_tensor(tensors: list[Tensor]):
            return any(t.origin.id == tensor.origin.id and t.memory_operand == tensor.memory_operand for t in tensors)

        instances_storing_related_tensor = [
            instance
            for instance, tensors in self.accelerator.memory_manager.top_instance_stored_tensors.items()
            if contains_related_tensor(tensors)
        ]
        return 1 / len(instances_storing_related_tensor)

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
