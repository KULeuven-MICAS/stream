from operator import attrgetter, itemgetter
from networkx import DiGraph
from stream.classes.cost_model.memory_manager import MemoryManager
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.tensor import Tensor
import logging

logger = logging.getLogger(__name__)


def schedule_graph(G: DiGraph, accelerator: Accelerator, cores_idle_from=None, candidate_selection='latency'):
    """Schedule the nodes of graph G across the cores in the system.
    Each node should have a core_allocation and runtime set.

    Args:
        G (DiGraph): Graph containing the nodes to be scheduled.
        accelerator (Accelerator): The accelerator to schedule the nodes on.
        cores_start_offset (dict, optional): A dict containing for each core_id its start offset. Defaults to None.
    """
    # Initialize total link energy cost and memory energy cost
    total_computation_energy_cost = 0
    total_link_energy_cost = 0
    total_memory_energy_cost = 0
    total_eviction_link_energy_cost = 0
    total_eviction_memory_energy_cost = 0
    input_onloading_link_energy_cost = 0
    input_onloading_memory_energy_cost = 0
    output_offloading_link_energy_cost = 0
    output_offloading_memory_energy_cost = 0

    all_core_ids = sorted(list(set(n.core_allocation for n in G.nodes())))
    if cores_idle_from is None:
        # Make it 0 for all cores
        cores_idle_from = {core_allocation: 0 for core_allocation in all_core_ids}
    last_node_start_time = {core_allocation: 0 for core_allocation in all_core_ids}
    # cores_memory_usage = {core_id: list() for core_id in all_core_ids}  # init buffer usage at different timesteps

    nb_graph_nodes = G.number_of_nodes()
    nb_scheduled_nodes = 0
    scheduled_nodes = set()

    # List that keeps all possible candidate nodes for each core.
    candidates = []

    # Get all the nodes with no predecessors and put them in the candidates queues for the core onto which they are mapped
    source_layers = sorted(set(n.id[0] for n, d in G.in_degree() if d == 0))
    source_layer_nodes = set((n for n in G.nodes() if n.id[0] in source_layers))

    # Put the very first nodes of a layer that doesn't have any incoming edges as the first candidates
    for source_node in (n for n, d in G.in_degree() if d == 0):
        core_allocation = source_node.core_allocation
        # core_candidates[core_allocation].append((cores_idle_from[core_allocation], source_node))
        candidates.append((cores_idle_from[core_allocation], source_node))

    # Get all the nodes with no successors that produce final outputs, used for off-loading final outputs
    sink_layers = sorted(set(n.id[0] for n, d in G.out_degree() if d == 0))
    sink_layer_nodes = set((n for n in G.nodes() if (n.id[0] in sink_layers) and (n.produces_final_output is True)))

    # Get the offchip core id
    offchip_core_id = accelerator.offchip_core_id

    ## Schedule preparation:
    # 1. Initialize the total and core priority for each tensor
    # 2. Add the constant operand tensors of all nodes to the off-chip initially
    for n in G.nodes():
        for op, tensor in n.operand_tensors.items():
            tensor.initialize_core_priorities(G, n)
            if op in n.constant_operands:
                if not accelerator.contains_tensor(tensor, offchip_core_id):
                    accelerator.memory_manager.add_tensor_to_core(
                        tensor=tensor,
                        core_id=offchip_core_id,
                        timestep=0,
                        timestep_end=0,
                        tensors_to_avoid_evicting=[])

    done = False
    logger.info("Start scheduling.")
    while not done:

        # If this core doesn't have any candidates, continue to the next core
        if not candidates:
            raise ValueError(f"There are no candidates to schedule and only {nb_scheduled_nodes}/{nb_graph_nodes} nodes have been scheduled.")
        if candidate_selection == "latency":
            # Get the best candidate: the one with the earliest possible start time
            (preds_end, best_candidate) = min(candidates)
        elif candidate_selection == "memory":
            # Get the best candidate: the one with the highest layer_id
            preds_ends, cn_candidates = zip(*candidates)
            best_candidate = max(cn_candidates, key=attrgetter('id'))
            preds_end = preds_ends[cn_candidates.index(best_candidate)]
        else:
            raise ValueError(f"Scheduler's CN candidate_selection criterion '{candidate_selection}' is not supported.")
        # Remove this candidate from the candidates (as we are going to schedule it)
        candidates.remove((preds_end, best_candidate))

        core_id = best_candidate.core_allocation
        core = accelerator.get_core(core_id)

        start = max(cores_idle_from[core_id], preds_end)  # init start time when the core becomes available

        ## Step 0
        # Determine all the tensors needed to compute this candidate
        # Constant operands
        tensors_operands = [best_candidate.memory_operand_links[op] for op in best_candidate.constant_operands]
        tensors_this_candidate_needs = [best_candidate.operand_tensors[op] for op in best_candidate.constant_operands]
        # Non-constant operands
        for (pred, best_candidate, edge_data) in sorted(G.in_edges(best_candidate, data=True), key=itemgetter(0)):
            if pred.id[0] == best_candidate.id[0]:
                continue  # Skip if predecessor was from the same layer (intra-edge)
            pred_output_tensor = pred.operand_tensors[pred.output_operand]
            tensors_this_candidate_needs.append(pred_output_tensor)
            tensors_operands.append(best_candidate.memory_operand_links[edge_data['operand']])
        # Sort these tensors based on their earliest possible transfer time
        tensors_this_candidate_needs, tensors_operands = zip(*sorted(zip(tensors_this_candidate_needs, tensors_operands)))

        ## Step 1
        # There could be operands that are too large to store in the highest memory on the core
        # The tensors stored in these memories should be evicted and potentially written back to off-chip
        # Clear these memories (this might delay the potential start time if things have to written to off-chip)
        timestep = start
        for too_large_operand in best_candidate.too_large_operands:
            eviction_link_energy_cost, eviction_memory_energy_cost, timestep = \
                accelerator.memory_manager.evict_all_tensors_from_core(core_id, too_large_operand, timestep, tensors_this_candidate_needs)
            total_eviction_link_energy_cost += eviction_link_energy_cost
            total_eviction_memory_energy_cost += eviction_memory_energy_cost

        ## Step 2
        # The computation might need tensors that are currently not present in the core's memories
        # We need to fetch these tensors from either off-chip or from the core where they are present
        # Transfer these tensors from wherever they are currently residing to this core
        for tensor, tensor_operand in zip(tensors_this_candidate_needs, tensors_operands):
            if tensor_operand not in best_candidate.too_large_operands:
                # Transfer the tensor
                worst_case_timestep = timestep
                transfer_complete_timestep, transfer_link_energy_cost, transfer_memory_energy_cost, eviction_link_energy_cost, eviction_memory_energy_cost = \
                    accelerator.transfer_tensor_to_core(tensor, core_id, tensor_operand, tensors_this_candidate_needs, worst_case_timestep)
                # Update the possible start time of this node
                timestep = max(timestep, transfer_complete_timestep)
                # Add the energy costs to their respective trackers
                total_link_energy_cost += transfer_link_energy_cost
                total_memory_energy_cost += transfer_memory_energy_cost
                total_eviction_link_energy_cost += eviction_link_energy_cost
                total_eviction_memory_energy_cost += eviction_memory_energy_cost

        ## Step 3
        # Check if we had any operands that were too large to store in the core's memory, block the relevant off-chip link for the duration
        # This might again delay the execution if the offchip link was already blocked by another core
        timestep = accelerator.block_offchip_links(best_candidate.too_large_operands, core_id, timestep, best_candidate.get_runtime(), best_candidate.id)
        # Get the start and end time of the candidate
        start = timestep
        end = start + best_candidate.get_runtime()

        ## Step 4
        # Add the output tensor of this computation node to the memory manager at the start time of the computation
        # If the output operand is in the too large operands, add it to off-chip, otherwise add it to this core's output memory
        output_layer_operand = best_candidate.output_operand
        output_memory_operand = best_candidate.memory_operand_links[output_layer_operand]
        output_tensor = best_candidate.operand_tensors[output_layer_operand]
        if output_memory_operand in best_candidate.too_large_operands:
            core_id_to_add_output_to = offchip_core_id
        else:
            core_id_to_add_output_to = core_id
        evictions_complete_timestep, eviction_link_energy_cost, eviction_memory_energy_cost = \
            accelerator.memory_manager.add_tensor_to_core(output_tensor, core_id_to_add_output_to, start, end, tensors_this_candidate_needs)
        total_eviction_link_energy_cost += eviction_link_energy_cost
        total_eviction_memory_energy_cost += eviction_memory_energy_cost
        start = evictions_complete_timestep
        end = start + best_candidate.get_runtime()

        ## Step 5
        # Update the start and end time of the node
        best_candidate.set_start(start)
        best_candidate.set_end(end)
        cores_idle_from[core_id] = end
        last_node_start_time[core_id] = start

        # Add the computation energy of running this node
        total_computation_energy_cost += best_candidate.energy

        # Add this node to the scheduled nodes
        scheduled_nodes.add(best_candidate)

        ## Step 6
        # Memory usage: When the node ends:
        # Decrease the priority of all the tensors this node used
        for (tensor_used_by_node, tensor_memory_operand) in zip(tensors_this_candidate_needs, tensors_operands):
            tensor_used_by_node.core_priorities[core_id] -= 1
            if tensor_used_by_node.get_total_priority() == 0:
                cores_storing_tensor_used_by_node, top_level_idxs, stored_since_timesteps = accelerator.memory_manager.find_tensor(tensor_used_by_node)
                for storing_core_id, top_level_idx in zip(cores_storing_tensor_used_by_node, top_level_idxs):
                    storing_core = accelerator.get_core(storing_core_id)
                    accelerator.memory_manager.remove_tensor_from_core(storing_core, top_level_idx, tensor_used_by_node, end, write_back_to_offchip=False)

        ## Step 7
        # Memory usage: When the node ends:
        # If this node is a sink node (node that has no successors and that produces a final output), transfer final outputs to offchip
        if best_candidate in sink_layer_nodes:
            top_level_idx = accelerator.memory_manager.get_top_level_idx(core, output_tensor.memory_operand)
            # Only push back sink node outputs if they're generated and stored on the core
            if not best_candidate.output_operand in best_candidate.too_large_operands:
                current_timestep, link_energy_cost, memory_energy_cost = \
                    accelerator.memory_manager.remove_tensor_from_core(core, top_level_idx, output_tensor, end, write_back_to_offchip=True)
                output_offloading_link_energy_cost += link_energy_cost
                output_offloading_memory_energy_cost += memory_energy_cost

        ## Step 8
        # For each successor of this node, check if all of its predecessors have been scheduled
        for successor in sorted(G.successors(best_candidate)):
            if all((pred in scheduled_nodes for pred in G.predecessors(successor))):
                preds_end = max((predecessor.end for predecessor in G.predecessors(successor)), default=0)
                # core_candidates[successor.core_allocation].append((preds_end, successor))
                candidates.append((preds_end, successor))

        # Increment the number of scheduled nodes
        nb_scheduled_nodes += 1
        done = nb_scheduled_nodes == nb_graph_nodes

    latency = max((n.end for n in G.nodes()))
    # print("Scheduling completed")
    # print(f"Latency found = {latency}")
    return latency, total_computation_energy_cost, total_memory_energy_cost, total_link_energy_cost, input_onloading_link_energy_cost, input_onloading_memory_energy_cost, output_offloading_link_energy_cost, output_offloading_memory_energy_cost, total_eviction_link_energy_cost, total_eviction_memory_energy_cost
