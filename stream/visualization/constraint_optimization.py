import json
import logging
from typing import TYPE_CHECKING

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.allocation import ALLOCATION_T
from stream.opt.allocation.constraint_optimization.utils import (
    compute_iterations_overlap,
    get_k_splits,
    get_node_latencies,
    get_node_start_timesteps,
    get_node_timesteps,
    get_resources,
    get_timestep_latencies,
    get_timesteps,
)
from stream.utils import CostModelEvaluationLUT
from stream.workload.onnx_workload import ComputationNodeWorkload

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def to_perfetto_json(
    workload: ComputationNodeWorkload,
    allocation: ALLOCATION_T,
    cost_lut: CostModelEvaluationLUT,
    accelerator: Accelerator,
    iterations: int,
    latency_attr: str,
    json_path: str,
):
    """
    Allocation is a list of tuples, with each tuple being of form (timestep, allocation, node_id). Allocation is a core.
    cost_lut is a CostModelEvaluationLUT storing for each node and each core the hardware performance.
    """
    k_splits = get_k_splits(allocation)
    timesteps = get_timesteps(allocation)
    resources = get_resources(allocation)
    nodes = set(n for n in workload.node_list if (n.id, n.sub_id) in k_splits)
    node_latencies = get_node_latencies(nodes, allocation, cost_lut, accelerator, k_splits, latency_attr)
    node_timesteps = get_node_timesteps(allocation)
    timestep_latencies = get_timestep_latencies(allocation, node_latencies, timesteps)
    starts = get_node_start_timesteps(k_splits, node_timesteps, timestep_latencies)
    total_timestep_latency = sum(timestep_latencies.values())
    cores = sorted(set(k[1] for k in starts))
    overlap = compute_iterations_overlap(timestep_latencies, node_timesteps, starts, total_timestep_latency, cores)
    offset = total_timestep_latency - overlap

    # Prepare JSON data for Perfetto
    perfetto_data = []

    # Add thread names (cores)
    for core in resources:
        thread_name_event = {
            "name": "thread_name",
            "ph": "M",
            "pid": "waco",
            "tid": core,
            "args": {"name": f"Core {core}"},
            "cname": "blue",
        }
        perfetto_data.append(thread_name_event)

    # Add events for each iteration
    for iteration in range(iterations):
        iteration_offset = iteration * offset
        for id, allocations in k_splits.items():
            for a in allocations:
                start = starts[id, a] + iteration_offset
                runtime = node_latencies[id, a]
                event = {
                    "name": f"Node {id}",
                    "cat": "compute",
                    "ph": "X",
                    "ts": start,
                    "dur": runtime,
                    "pid": "waco",
                    "tid": a,
                    "cname": "blue",
                    "args": {"Runtime": runtime, "NodeID": id, "Iteration": iteration},
                }
                perfetto_data.append(event)

    # Write JSON data to file
    with open(json_path, "w") as f:
        json.dump(perfetto_data, f, indent=2)

    logger.info(f"Plotted WACO result to {json_path}")
