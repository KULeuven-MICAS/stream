from math import prod
from typing import TYPE_CHECKING

from zigzag.datatypes import LayerDim, LayerOperand, UnrollFactor

from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.allocation import ALLOCATION_T

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
    for slot, core, k in ids_list:
        new_l.append((slot, core, invert_id(k)))
    return new_l


def get_loop_size(loops: list[tuple[LayerDim, UnrollFactor]], dims: list[LayerDim]) -> int:
    return int(prod([tl[1] for tl in loops if tl[0] in dims]))


def get_latencies(
    nodes: list[ComputationNode],
    core_ids: list[int],
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    impossible_lat: float = 1e11,
    ids: dict[ComputationNode, int] = {},
    latency_attr: str = "latency_total1",
) -> tuple[dict[tuple[int, str, int], int], dict]:
    if not ids:
        ids = {node: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    latencies = {(ids[node], core_name): impossible_lat for node in nodes for core_name in core_names}
    possible_allocations: dict[int, list[str]] = {}
    inter_core_tiling_sizes = {}

    for node in nodes:
        node_id = ids[node]
        possible_allocations[node_id] = []
        for core_id, core_name in zip(core_ids, core_names):
            core = accelerator.get_core(core_id)
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
                inter_core_tiling_sizes[(node_id, core_name)] = inter_core_tiling_size
                lat = getattr(cme, latency_attr)
                possible_allocations[node_id].append(core_name)
            except ValueError:
                lat = impossible_lat
            latencies[(node_id, core_name)] = lat

    latencies_with_split = {}
    possible_allocation_splits = {}
    p_max = len(core_names)  # maximum parallelization factor

    for node_id in ids.values():
        possible_allocation_splits[node_id] = {}
        for core_name in core_names:
            possible_allocation_splits[node_id][core_name] = {}
            if core_name in possible_allocations[node_id]:
                p_t = int(inter_core_tiling_sizes[node_id, core_name])
                for p in range(1, p_max + 1):
                    if p <= len(possible_allocations[node_id]):
                        lat = int(latencies[(node_id, core_name)] / min(p_t, p))
                        possible_allocation_splits[node_id][core_name][p] = 1
                    else:
                        lat = impossible_lat
                        possible_allocation_splits[node_id][core_name][p] = 0
                    latencies_with_split[(node_id, core_name, p)] = lat
            else:
                for p in range(1, p_max + 1):
                    latencies_with_split[(node_id, core_name, p)] = impossible_lat
                    possible_allocation_splits[node_id][core_name][p] = 0

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


def get_k_splits(allocation: "ALLOCATION_T") -> dict[tuple[int, int], list[str]]:
    k_splits: dict[tuple[int, int], list[str]] = {}
    for _, core, id in allocation:
        k_splits[id] = k_splits.get(id, []) + [core]
    return k_splits


def get_node_latencies(
    nodes: set[ComputationNode],
    allocation: "ALLOCATION_T",
    cost_lut: CostModelEvaluationLUT,
    accelerator: Accelerator,
    k_splits: dict[tuple[int, int], list[str]],
    latency_attr: str,
) -> dict[tuple[tuple[int, int], str], int]:
    node_latencies: dict[tuple[tuple[int, int], str], int] = {}
    core_names = sorted(set([a for _, a, _ in allocation]))
    core_ids = [int(core_name.split(" ")[-1]) for core_name in core_names]
    for _, a, combined_id in allocation:
        layer_id, _ = combined_id
        node = next(node for node in nodes if node.id == layer_id)
        latencies, _ = get_latencies([node], core_ids, accelerator, cost_lut, latency_attr=latency_attr)
        nb_k_splits = len(k_splits[combined_id])
        lat = latencies[(node.id, a, nb_k_splits)]
        node_latencies[combined_id, a] = lat
    return node_latencies


def get_layer_ids(allocation: "ALLOCATION_T") -> list[int]:
    layer_ids_set: set[int] = set()
    for _, _, id in allocation:
        layer_ids_set.add(id[0])
    layer_ids = sorted(layer_ids_set)
    return layer_ids


def get_timesteps(allocation: "ALLOCATION_T") -> list[int]:
    return [item[0] for item in allocation]


def get_resources(allocation: "ALLOCATION_T") -> set[str]:
    return set(item[1] for item in allocation)


def get_node_timesteps(allocation: "ALLOCATION_T") -> dict[tuple[tuple[int, int], str], int]:
    node_timesteps: dict[tuple[tuple[int, int], str], int] = {}
    for t, a, id in allocation:
        node_timesteps[id, a] = t
    return node_timesteps


def get_timestep_latencies(
    allocation: "ALLOCATION_T", node_latencies: dict[tuple[tuple[int, int], str], int], timesteps: list[int]
) -> dict[int, int]:
    timestep_latencies = {t: 0 for t in range(max(timesteps) + 1)}
    for t, a, id in allocation:
        timestep_latencies[t] = max(timestep_latencies.get(t, 0), node_latencies[id, a])
    return timestep_latencies


def get_node_start_timesteps(
    k_splits: dict[tuple[int, int], list[str]],
    node_timesteps: dict[tuple[tuple[int, int], str], int],
    timestep_latencies: dict[int, int],
):
    starts = {}
    for id, allocations in k_splits.items():
        for a in allocations:
            start = get_start_time_of_node(id, a, node_timesteps, timestep_latencies)
            starts[id, a] = start
    return starts


def get_start_time_of_node(id, a, timesteps, timestep_latencies, t_start=0):
    node_timestep = timesteps[id, a]
    for t in range(node_timestep):
        t_end = t_start + timestep_latencies[t]
        t_start = t_end
    return t_start


def calculate_total_latency(
    nodes: set[ComputationNode],
    allocation: "ALLOCATION_T",
    cost_lut: CostModelEvaluationLUT,
    accelerator: Accelerator,
    iterations: int,
    latency_attr: str,
) -> tuple[int, str]:
    k_splits = get_k_splits(allocation)
    timesteps = get_timesteps(allocation)
    node_latencies = get_node_latencies(nodes, allocation, cost_lut, accelerator, k_splits, latency_attr)
    timestep_latencies = get_timestep_latencies(allocation, node_latencies, timesteps)
    node_timesteps = get_node_timesteps(allocation)
    starts = get_node_start_timesteps(k_splits, node_timesteps, timestep_latencies)
    total_timestep_latency = sum(timestep_latencies.values())
    cores = sorted(set(k[1] for k in starts))
    overlap = compute_iterations_overlap(timestep_latencies, node_timesteps, starts, total_timestep_latency, cores)
    total_lat = iterations * total_timestep_latency - (iterations - 1) * overlap
    total_lat_str = f"total_lat = N * T - (N - 1) * overlap --> {total_lat} = {iterations} * {total_timestep_latency} - {iterations-1} * {overlap}"
    return total_lat, total_lat_str


def compute_iterations_overlap(timestep_latencies, node_timesteps, starts, T, cores):
    slacks = {}
    for core in cores:
        relevant_starts = [v for k, v in starts.items() if k[1] == core]
        earliest_start = min(relevant_starts)
        latest_start = max(relevant_starts)
        latest_id_core = next((k for k, v in starts.items() if v == latest_start and k[1] == core))
        latest_timestep = node_timesteps[latest_id_core]
        timestep_latency = timestep_latencies[latest_timestep]
        latest_end = latest_start + timestep_latency
        slack = T - latest_end + earliest_start
        assert slack >= 0
        slacks[core] = slack
    overlap = min(slacks.values())
    return overlap
