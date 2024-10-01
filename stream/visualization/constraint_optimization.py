import logging
import os
from itertools import cycle

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.utils import get_latencies
from stream.utils import CostModelEvaluationLUT

logger = logging.getLogger(__name__)


def visualize_waco(
    allocation, node_hw_performances: CostModelEvaluationLUT, accelerator: Accelerator, fig_path: str, iterations: int
):
    """
    Allocation is a list of tuples, with each tuple being of form (timestep, allocation, node_id). Allocation is a core.
    node_hw_performances is a nested dict storing for each node and each core the hardware performance.
    """
    pass
    # Extract the number of allocations (k splits) of all nodes
    k_splits = {}
    for _, a, id in allocation:
        k_splits[id] = k_splits.get(id, []) + [a]
    # Extract the latencies of all nodes
    node_latencies = {}
    layer_ids = set()
    ids = []
    timesteps = []
    resources = set()
    core_names = sorted(set([a for t, a, id in allocation]))
    core_ids = [int(core_name.split(" ")[-1]) for core_name in core_names]
    for t, a, id in allocation:
        timesteps.append(t)
        layer_ids.add(id[0])
        ids.append(id)
        resources.add(a)
        node = next(n for n in node_hw_performances.get_nodes() if n.id == id[0])
        latencies, _ = get_latencies([node], core_ids, accelerator, node_hw_performances)
        nb_k_splits = len(k_splits[id])
        lat = latencies[(node.id, a, nb_k_splits)]
        node_latencies[id, a] = lat
    layer_ids = sorted(layer_ids)
    # Extract the timesteps of all nodes and the latency per timestep
    node_timesteps = {}
    timestep_latencies = {t: 0 for t in range(max(timesteps) + 1)}
    for t, a, id in allocation:
        node_timesteps[id, a] = t
        timestep_latencies[t] = max(timestep_latencies.get(t, 0), node_latencies[id, a])
    # Extract start of each node
    starts = {}
    for id, allocations in k_splits.items():
        for a in allocations:
            start = get_start_time_of_node(id, a, node_timesteps, timestep_latencies)
            starts[id, a] = start
    _ = calculate_total_latency(starts, timestep_latencies, node_timesteps, iterations)
    # Plot the nodes using Plotly rectangles
    color_cycle = cycle(sample_colorscale("rainbow", np.linspace(0, 1, len(node_hw_performances.get_nodes()))))
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
    fig.update_layout(title_text="Constraint optimization timeslot visualization")

    fig.update_yaxes(categoryorder="array", categoryarray=sorted(resources))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="stack")

    # Create subfolders if they don't exist
    dir_name = os.path.dirname(os.path.abspath(fig_path))
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    fig.write_html(fig_path)
    logger.info(f"Plotted WACO result to {fig_path}")


def get_start_time_of_node(id, a, timesteps, timestep_latencies, t_start=0):
    node_timestep = timesteps[id, a]
    for t in range(node_timestep):
        t_end = t_start + timestep_latencies[t]
        t_start = t_end
    return t_start


def calculate_total_latency(starts, timestep_latencies, node_timesteps, N):
    T = sum(timestep_latencies.values())
    cores = sorted(set(k[1] for k in starts))
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
    min_slack = min(slacks.values())
    total_lat = N * T - (N - 1) * min_slack
    print("total_lat = N * T - (N - 1) * slack")
    print(f"{total_lat} = {N} * {T} - {N-1} * {min_slack}")
    return total_lat
