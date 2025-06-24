from math import prod
from typing import TYPE_CHECKING

from zigzag.datatypes import LayerDim, LayerOperand, UnrollFactor

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.allocation import TimeSlotAllocation

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


def invert_ids_list(ids_list: list[tuple[int, int, int]], nb_nodes: int) -> list[tuple[int, int, tuple[int, int]]]:
    new_l: list[tuple[int, int, tuple[int, int]]] = []
    for timestep, core, k in ids_list:
        new_l.append((timestep, core, invert_id(k)))
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
    ids: dict[ComputationNode, int] = {},
) -> dict[tuple[int, str], float]:
    if not ids:
        ids = {node.id: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    energies = {(ids[node], core_name): impossible_energy for node in nodes for core_name in core_names}

    for node in nodes:
        for core_id, core_name in zip(core_ids, core_names):
            core = accelerator.get_core(core_id)
            try:
                cme = cost_lut.get_cme(node, core)
                en = getattr(cme, "energy_total")
            except ValueError:
                en = impossible_energy
            energies[(ids[node], core_name)] = en

    return energies


def get_node_latencies(allocation: "TimeSlotAllocation", cost_lut, accelerator, latency_attr):
    node_latencies = {}
    for node in allocation.nodes:
        cores = allocation.get_cores_for_node(node)
        core_ids = [core.id for core in cores]
        latencies, _ = get_latencies([node], core_ids, accelerator, cost_lut, latency_attr=latency_attr)
        p = len(cores)
        for core in cores:
            combined_id = (node, core, p)
            lat = latencies[combined_id]
            node_latencies[node, core] = lat
    return node_latencies


def get_layer_ids(allocation: "TimeSlotAllocation"):
    return [n.id for n in allocation.nodes]


def get_timesteps(allocation: "TimeSlotAllocation") -> list[int]:
    return allocation.slots


def get_resources(allocation: "TimeSlotAllocation") -> list[int]:
    return [core.id for core in allocation.cores]


def get_timestep_latencies(allocation: "TimeSlotAllocation", node_latencies, timesteps):
    timestep_latencies = {t: 0 for t in range(max(timesteps) + 1)}
    for timestep in timesteps:
        max_latency = 0
        for core, node in allocation.get_allocations_in_slot(timestep).items():
            id = (node, core)
            if id in node_latencies:
                max_latency = max(max_latency, node_latencies[id])
            else:
                raise ValueError(f"Node {node} on core {core} not found in node latencies.")
        timestep_latencies[timestep] = max_latency
    return timestep_latencies


def get_node_start_timesteps(allocation: "TimeSlotAllocation", timestep_latencies):
    starts = {}
    for timestep, core, node in allocation.allocations:
        start = get_start_time_of_node(timestep, timestep_latencies)
        starts[node, core] = start
    return starts


def get_start_time_of_node(timestep, timestep_latencies, t_start=0):
    for t in range(timestep):
        t_end = t_start + timestep_latencies[t]
        t_start = t_end
    return t_start


def calculate_total_latency(allocation, cost_lut, accelerator, iterations, latency_attr) -> tuple[int, str]:
    timesteps = get_timesteps(allocation)
    node_latencies = get_node_latencies(allocation, cost_lut, accelerator, latency_attr)
    timestep_latencies = get_timestep_latencies(allocation, node_latencies, timesteps)
    starts = get_node_start_timesteps(allocation, timestep_latencies)
    total_timestep_latency = sum(timestep_latencies.values())
    overlap = compute_iterations_overlap(allocation, timestep_latencies, starts, total_timestep_latency)
    total_lat = iterations * total_timestep_latency - (iterations - 1) * overlap
    total_lat_str = f"total_lat = N * T - (N - 1) * overlap --> {total_lat} = {iterations} * {total_timestep_latency} - {iterations-1} * {overlap}"
    return total_lat, total_lat_str


def compute_iterations_overlap(allocation: "TimeSlotAllocation", timestep_latencies, starts, T):
    slacks = {}
    for core in allocation.cores:
        relevant_starts = [v for k, v in starts.items() if k[1] == core]
        earliest_start = min(relevant_starts)
        latest_start = max(relevant_starts)
        latest_node_on_this_core = next((k for k, v in starts.items() if v == latest_start and k[1] == core))[0]
        latest_timestep = allocation.get_timeslot_of_node_on_core(latest_node_on_this_core, core)
        timestep_latency = timestep_latencies[latest_timestep]
        latest_end = latest_start + timestep_latency
        slack = T - latest_end + earliest_start
        assert slack >= 0
        slacks[core] = slack
    overlap = min(slacks.values())
    return overlap
