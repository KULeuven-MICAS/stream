from math import ceil, prod
from typing import TYPE_CHECKING

from zigzag.datatypes import LayerDim, LayerOperand, UnrollFactor

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.steady_state.computation import SteadyStateComputation

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.timeslot_allocation import TimeSlotAllocation

MODULATION_NUMBER = 1 << 20  # Must be higher than any node's sub id


def convert_id(i: int, j: int) -> int:
    assert 0 <= j < MODULATION_NUMBER
    return MODULATION_NUMBER * i + j


def invert_id(k: int) -> tuple[int, int]:
    i, j = divmod(k, MODULATION_NUMBER)
    return i, j


def convert_ids(nodes: list[ComputationNode]):
    ids: dict[ComputationNode, int] = {}
    for node in nodes:
        i, j = node.id, node.sub_id
        new_id = convert_id(i, j)
        ids[node] = new_id
    return ids


def invert_ids_list(ids_list: list[tuple[int, str, int]], nb_nodes: int) -> list[tuple[int, int, tuple[int, int]]]:
    new_l: list[tuple[int, int, tuple[int, int]]] = []
    for timestep, core_str, k in ids_list:
        core_id = int(core_str.split(" ")[-1])  # Extract core id from "Core <id>"
        new_l.append((timestep, core_id, invert_id(k)))
    return new_l


def get_loop_size(loops: list[tuple[LayerDim, UnrollFactor]], dims: list[LayerDim]) -> int:
    return int(prod([tl[1] for tl in loops if tl[0] in dims]))


def get_latencies(
    nodes: list[ComputationNode],
    core_ids: list[int],
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    impossible_lat: float = 1e11,
    latency_attr: str = "latency_total1",
) -> tuple[dict[tuple[ComputationNode, Core, int], int], dict]:
    cores = [accelerator.get_core(core_id) for core_id in core_ids]
    latencies = {(node, core): impossible_lat for node in nodes for core in cores}
    possible_allocations: dict[ComputationNode, list[Core]] = {}
    inter_core_tiling_sizes = {}

    for node in nodes:
        possible_allocations[node] = []
        for core in cores:
            try:
                equal_node = cost_lut.get_equal_node(node)
                assert equal_node, f"No equal node for {node} found in CostModelEvaluationLUT"
                cme = cost_lut.get_cme(equal_node, core)
                output_operand = LayerOperand("O")
                temporal_loops = [
                    i for tm_level in cme.temporal_mapping.mapping_dic_stationary[output_operand] for i in tm_level
                ]
                inter_core_tiling_dims = [layer_dim for layer_dim, _ in node.inter_core_tiling]
                inter_core_tiling_size = get_loop_size(temporal_loops, inter_core_tiling_dims)
                inter_core_tiling_sizes[(node, core)] = inter_core_tiling_size
                lat = getattr(cme, latency_attr)
                possible_allocations[node].append(core)
            except ValueError:
                lat = impossible_lat
            latencies[(node, core)] = lat

    latencies_with_split = {}
    possible_allocation_splits = {}
    p_max = len(cores)  # maximum parallelization factor

    for node in nodes:
        possible_allocation_splits[node] = {}
        for core in cores:
            possible_allocation_splits[node][core] = {}
            if core in possible_allocations[node]:
                p_t = int(inter_core_tiling_sizes[node, core])
                for p in range(1, p_max + 1):
                    if p <= len(possible_allocations[node]):
                        lat = int(latencies[(node, core)] / min(p_t, p))
                        possible_allocation_splits[node][core][p] = 1
                    else:
                        lat = impossible_lat
                        possible_allocation_splits[node][core][p] = 0
                    latencies_with_split[(node, core, p)] = lat
            else:
                for p in range(1, p_max + 1):
                    latencies_with_split[(node, core, p)] = impossible_lat
                    possible_allocation_splits[node][core][p] = 0

    return latencies_with_split, possible_allocation_splits


def get_energies(
    nodes: list[ComputationNode],
    core_ids: list[int],
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    impossible_energy: float = 1e11,
    ids: dict[ComputationNode, int] | None = None,
) -> dict[tuple[int, str], float]:
    if ids is None:
        ids = {node: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    energies = {(ids[node], core_name): impossible_energy for node in nodes for core_name in core_names}

    for node in nodes:
        for core_id, core_name in zip(core_ids, core_names, strict=False):
            core = accelerator.get_core(core_id)
            try:
                cme = cost_lut.get_cme(node, core)
                en = cme.energy_total
            except ValueError:
                en = impossible_energy
            energies[(ids[node], core_name)] = en

    return energies


def get_node_latencies(allocation: "TimeSlotAllocation", cost_lut, accelerator, latency_attr):
    node_latencies = {}
    for node in allocation.get_computation_nodes():
        cores = allocation.get_resources_for_node(node)
        assert all(isinstance(core, Core) for core in cores), f"Node {node} has non-core resources: {cores}"
        core_ids = [core.id for core in cores if isinstance(core, Core)]
        latencies, _ = get_latencies([node], core_ids, accelerator, cost_lut, latency_attr=latency_attr)
        p = len(cores)
        for core in cores:
            assert isinstance(core, Core), f"Resource {core} for node {node} is not a Core."
            combined_id = (node, core, p)
            lat = latencies[combined_id]
            node_latencies[node, core] = lat
    return node_latencies


def get_timestep_latencies(allocation: "TimeSlotAllocation", timesteps):
    timestep_latencies = {t: 0 for t in range(max(timesteps) + 1)}
    for timestep in timesteps:
        max_latency = 0
        for _, node in allocation.get_allocations_in_slot(timestep).items():
            assert node.runtime is not None, f"Node {node} has no runtime set."
            max_latency = max(max_latency, int(node.runtime))
        timestep_latencies[timestep] = max_latency
    return timestep_latencies


def get_node_start_timesteps(allocation: "TimeSlotAllocation", timestep_latencies):
    starts = {}
    for timestep, core, node in allocation.allocations:
        start = get_start_time_of_timestep(timestep, timestep_latencies)
        starts[node, core] = start
    return starts


def get_start_time_of_timestep(timestep, timestep_latencies, t_start=0):
    for t in range(timestep):
        t_end = t_start + timestep_latencies[t]
        t_start = t_end
    return t_start


def calculate_total_latency(allocation: "TimeSlotAllocation", iterations) -> tuple[int, str]:
    timesteps = allocation.slots
    timestep_latencies = get_timestep_latencies(allocation, timesteps)
    starts = get_node_start_timesteps(allocation, timestep_latencies)
    total_timestep_latency = sum(timestep_latencies.values())
    overlap = compute_iterations_overlap(allocation, timestep_latencies, starts, total_timestep_latency)
    total_lat = iterations * total_timestep_latency - (iterations - 1) * overlap
    total_lat_str = (
        "total_lat = N * T - (N - 1) * overlap --> "
        f"{total_lat} = {iterations} * {total_timestep_latency} - {iterations - 1} * {overlap}"
    )
    return total_lat, total_lat_str


def compute_iterations_overlap(allocation: "TimeSlotAllocation", timestep_latencies, starts, iteration_latency):
    slacks = {}
    for core in allocation.resources:
        relevant_starts = [v for k, v in starts.items() if k[1] == core]
        earliest_start = min(relevant_starts)
        latest_start = max(relevant_starts)
        latest_node_on_this_core = next((k for k, v in starts.items() if v == latest_start and k[1] == core))[0]
        latest_timestep = allocation.get_timeslot_of_node_on_resource(latest_node_on_this_core, core)
        timestep_latency = timestep_latencies[latest_timestep]
        latest_end = latest_start + timestep_latency
        slack = iteration_latency - latest_end + earliest_start
        assert slack >= 0
        slacks[core] = slack
    overlap = min(slacks.values())
    return overlap


def get_partitioned_nodes(
    node: ComputationNode,
    core_allocations: list[Core],
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
) -> list[SteadyStateComputation]:
    """
    Get the partitioned SteadyStateComputation nodes for a given ComputationNode based on the core allocations.
    """
    # If the node is not partitioned, return it as is
    if len(core_allocations) == 1:
        core = next(iter(core_allocations))
        latencies, _ = get_latencies(
            [
                node,
            ],
            [
                core.id,
            ],
            accelerator,
            cost_lut,
        )
        runtime = latencies[(node, core, 1)]
        mapping_attr = node.extract_inter_core_mapping_attr()
        possible_resource_allocation = [
            core,
        ]
        new_node = SteadyStateComputation(
            id=node.id,
            sub_id=node.sub_id,
            node_name=node.name,
            node_attr=node.extract_node_attr(),
            mapping_attr=mapping_attr,
            operand_tensor_reshape=node.operand_tensor_reshape,
            produces_final_output=node.produces_final_output,
            group_id=node.group,
            input_names=node.input_names,
            partially_constant_operands=node.partially_constant_operands,
            possible_resource_allocation=possible_resource_allocation,
        )
        new_node.set_runtime(runtime)
        return [
            new_node,
        ]
    # Get the inter-core-tiling
    inter_core_tiling = node.inter_core_tiling
    if len(inter_core_tiling) > 1:
        raise NotImplementedError(
            f"Partitioning of nodes with inter-core tiling {inter_core_tiling} is not supported yet."
        )
    tiling_dim = inter_core_tiling[0][0]
    original_size = node.layer_dim_sizes[tiling_dim]
    nb_tiles = len(core_allocations)
    # Get the original loop range of the node for the tiling dimension
    original_loop_range = node.loop_ranges[tiling_dim]
    # Calculate the new loop range for each partitioned node
    partitioned_loop_ranges = [
        (
            original_loop_range[0] + i * (original_loop_range[1] // nb_tiles),
            original_loop_range[0] + (i + 1) * (original_loop_range[1] // nb_tiles),
        )
        for i in range(nb_tiles)
    ]
    size_per_tile = ceil(original_size / nb_tiles)
    partitioned_nodes: list[SteadyStateComputation] = []
    latencies, _ = get_latencies(
        [
            node,
        ],
        [core.id for core in core_allocations],
        accelerator,
        cost_lut,
    )
    runtimes = {core: latencies[(node, core, len(core_allocations))] for core in core_allocations}
    for i, core in enumerate(core_allocations):
        # Update the layer_dim_sizes for the smaller partitioned tile
        node_attr = node.extract_node_attr()
        node_attr.layer_dim_sizes[tiling_dim] = size_per_tile
        inter_core_mapping_attr = node.extract_inter_core_mapping_attr()
        inter_core_mapping_attr.inter_core_tiling = [(tiling_dim, nb_tiles)]
        possible_resource_allocation = [
            core,
        ]
        # Create a new ComputationNode for each core allocation
        partitioned_node = SteadyStateComputation(
            id=node.id,
            sub_id=node.sub_id,
            node_name=node.name + f".part{i}",
            node_attr=node_attr,
            mapping_attr=inter_core_mapping_attr,
            operand_tensor_reshape=node.operand_tensor_reshape,
            produces_final_output=node.produces_final_output,
            group_id=node.group,
            input_names=node.input_names,
            partially_constant_operands=node.partially_constant_operands,
            possible_resource_allocation=possible_resource_allocation,
        )
        partitioned_node.loop_ranges[tiling_dim] = partitioned_loop_ranges[i]
        partitioned_node.set_chosen_core_allocation(core.id)
        partitioned_node.set_runtime(runtimes[core])
        partitioned_nodes.append(partitioned_node)
    return partitioned_nodes
