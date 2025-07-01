import json
import logging
from typing import TYPE_CHECKING

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.timeslot_allocation import TimeSlotAllocation
from stream.opt.allocation.constraint_optimization.utils import (
    get_node_latencies,
    get_node_start_timesteps,
    get_resources,
    get_timestep_latencies,
    get_timesteps,
)
from stream.utils import CostModelEvaluationLUT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def to_perfetto_json(
    allocation: TimeSlotAllocation,
    cost_lut: CostModelEvaluationLUT,
    accelerator: Accelerator,
    latency_attr: str,
    json_path: str,
):
    """
    Allocation is a list of tuples, with each tuple being of form (timestep, allocation, node_id). Allocation is a core.
    cost_lut is a CostModelEvaluationLUT storing for each node and each core the hardware performance.
    """
    timesteps = get_timesteps(allocation)
    resources = get_resources(allocation)
    node_latencies = get_node_latencies(allocation, cost_lut, accelerator, latency_attr)
    timestep_latencies = get_timestep_latencies(allocation, node_latencies, timesteps)
    starts = get_node_start_timesteps(allocation, timestep_latencies)
    # total_timestep_latency = sum(timestep_latencies.values())
    # overlap = compute_iterations_overlap(allocation, timestep_latencies, starts, total_timestep_latency)
    # offset = total_timestep_latency - overlap

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
    # for iteration in range(iterations):
    #     iteration_offset = iteration * offset
    for slot in range(allocation.slot_min, allocation.slot_max + 1):
        for core, node in allocation.get_allocations_in_slot(slot).items():
            start = starts[node, core]  # + iteration_offset
            runtime = node_latencies[node, core]
            event = {
                "name": f"{node}",
                "cat": "compute",
                "ph": "X",
                "ts": start,
                "dur": runtime,
                "pid": "waco",
                "tid": core.id,
                "cname": "blue",
                "args": {"Runtime": runtime, "Id": node.id, "Sub_id": node.sub_id},
            }
            perfetto_data.append(event)

    # Write JSON data to file
    with open(json_path, "w") as f:
        json.dump(perfetto_data, f, indent=2)

    logger.info(f"Plotted WACO result to {json_path}")
