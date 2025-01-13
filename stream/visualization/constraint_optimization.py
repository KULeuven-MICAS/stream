import json
import logging
import os
from itertools import cycle
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.allocation import ALLOCATION_T
from stream.opt.allocation.constraint_optimization.utils import (
    calculate_total_latency,
    compute_iterations_overlap,
    get_k_splits,
    get_layer_ids,
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


def visualize_waco(
    workload: ComputationNodeWorkload,
    allocation: ALLOCATION_T,
    cost_lut: CostModelEvaluationLUT,
    accelerator: Accelerator,
    iterations: int,
    latency_attr: str,
    fig_path: str,
):
    """
    Allocation is a list of tuples, with each tuple being of form (timestep, allocation, node_id). Allocation is a core.
    cost_lut is a CostModelEvaluationLUT storing for each node and each core the hardware performance.
    """
    k_splits = get_k_splits(allocation)
    layer_ids = get_layer_ids(allocation)
    timesteps = get_timesteps(allocation)
    resources = get_resources(allocation)
    node_timesteps = get_node_timesteps(allocation)
    node_latencies = get_node_latencies(workload, allocation, cost_lut, accelerator, k_splits, latency_attr)
    timestep_latencies = get_timestep_latencies(allocation, node_latencies, timesteps)
    starts = get_node_start_timesteps(k_splits, node_timesteps, timestep_latencies)
    _, total_lat_str = calculate_total_latency(allocation, cost_lut, accelerator, iterations, latency_attr)
    # Plot the nodes using Plotly rectangles
    color_cycle = cycle(sample_colorscale("rainbow", np.linspace(0, 1, len(cost_lut.get_nodes()))))
    colors = {layer_id: c for (layer_id, c) in zip(layer_ids, color_cycle)}
    fig = go.Figure()
    bars = []
    for id, allocations in k_splits.items():
        for a in allocations:
            start = starts[id, a]
            runtime = node_latencies[id, a]
            end = start + runtime
            resource = a
            name = f"Node {id}"
            layer_id = id[0]
            color = colors[layer_id]
            marker = {"color": color}
            hovertext = (
                f"<b>Task:</b> {name}<br><b>Start:</b> {start}<br><b>Runtime:</b> {runtime:.2e}<br><b>End:</b> {end}"
            )
            bar = go.Bar(
                base=[start],
                x=[runtime],
                y=[resource],
                orientation="h",
                name=name,
                marker=marker,
                hovertext=[hovertext],
                hoverinfo="text",
            )
            fig.add_trace(bar)
            bars.append(bar)

    # Vertical lines to indicate timeslots
    t_start = 0
    for timestep in range(max(timesteps) + 2):
        fig.add_vline(
            x=t_start,
            line_width=1,
            line_dash="dash",
        )
        t_start += timestep_latencies.get(timestep, 0)

    # Title
    fig.update_layout(title_text=total_lat_str)

    fig.update_yaxes(categoryorder="array", categoryarray=sorted(resources))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="stack")

    # Create subfolders if they don't exist
    dir_name = os.path.dirname(os.path.abspath(fig_path))
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    fig.write_html(fig_path)
    logger.info(f"Plotted WACO result to {fig_path}")


def to_perfetto_json(
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
    node_latencies = get_node_latencies(allocation, cost_lut, accelerator, k_splits, latency_attr)
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
