import argparse
import logging
import pickle
from collections import defaultdict
from itertools import cycle
from math import isnan
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from brokenaxes import brokenaxes
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from plotly.express.colors import sample_colorscale
from zigzag.datatypes import LayerOperand

from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import Tensor

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)


# MPL FONT SIZES
SMALLER_SIZE = 11
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 20

PLOT_DEPENDENCY_LINES_SAME_CORE = True

PLOTLY_HATCH_TYPES = {
    "compute": "",
    "block": "x",
    "transfer": "-",
}


def plot_timeline_brokenaxes(
    scme: "StreamCostModelEvaluation",
    draw_dependencies: bool = True,
    section_start_percent: tuple[int, ...] = (0, 50, 95),
    percent_shown: tuple[int, ...] = (5, 5, 5),
    plot_data_transfer: bool = False,
    fig_path: str = "outputs/schedule_plot.png",
) -> None:
    G: ONNXWorkload = scme.workload
    accelerator: Accelerator = scme.accelerator

    nb_layers = len(set(iter([n.id for n in G.node_list])))
    nb_cores = accelerator.cores.number_of_nodes()

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    if nb_layers > 6:
        plt.rc("legend", fontsize=SMALLER_SIZE)  # legend fontsize
    else:
        plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    logger.info("Plotting...")

    dep_linewidth = 1
    ANNOTATE_CN_ID = True
    tick_rotation = 0
    assert len(section_start_percent) == len(percent_shown)

    # Total latency of the SCME
    latency = scme.latency
    # Total energy of the SCME
    energy = scme.energy
    # total EDP of the SCME
    edp = latency * energy

    plt.figure(figsize=(20, 6))

    x_starts = [int((start / 100) * latency) for start in section_start_percent]
    x_ends = [
        int(((start + percent) / 100) * latency) for (start, percent) in zip(section_start_percent, percent_shown)
    ]

    bax = brokenaxes(xlims=tuple(zip(x_starts, x_ends)), wspace=0.05, d=0.005)
    axs = bax.axs
    nb_axs = len(axs)

    # colors = ["#233D4D", "#FE7F2D", "#FCCA46", "#A1C181", "#619B8A", "#BF0603", "#F4D58D", "#708D81", "#1DD3B0"]
    layer_colors = iter(plt.cm.rainbow(np.linspace(0, 1, nb_layers)))
    colors_seen = []

    handles = []
    layer_ids_seen = []
    legend_labels = []
    height = 0.5
    for cn in G.node_list:
        layer_id = cn.id
        # Get the colour for this layer
        if layer_id not in layer_ids_seen:
            color = next(layer_colors)
            colors_seen.append(color)
        else:
            color = colors_seen[layer_ids_seen.index(layer_id)]
        x = cn.start  # + 0.05
        y = cn.chosen_core_allocation - 0.25
        width = cn.runtime  # - 0.05
        if (
            (x_starts[0] <= x <= x_ends[0])
            or (x_starts[0] <= x + width <= x_ends[0])
            or (x_starts[0] > x and x + width > x_ends[0])
        ):
            # First ax is done separately because of handle
            handle = axs[0].add_patch(
                Rectangle(
                    xy=(x, y),
                    width=width,
                    height=height,
                    facecolor=color,
                    edgecolor="black",
                    lw=1,
                    label=f"Layer {layer_id}",
                )
            )
            if ANNOTATE_CN_ID:
                axs[0].annotate(
                    f"{cn.sub_id}",
                    (x + width / 2, y + 0.25),
                    color="black",
                    weight="bold",
                    fontsize=10,
                    ha="center",
                    va="center",
                )
        else:
            handle = None
        for ax_idx in range(1, nb_axs):
            if (
                (x_starts[ax_idx] <= x <= x_ends[ax_idx])
                or (x_starts[ax_idx] <= x + width <= x_ends[ax_idx])
                or (x_starts[ax_idx] > x and x + width > x_ends[ax_idx])
            ):
                axs[ax_idx].add_patch(
                    Rectangle(
                        xy=(x, y),
                        width=width,
                        height=height,
                        facecolor=color,
                        edgecolor="black",
                        lw=1,
                        label=f"Layer {layer_id}",
                    )
                )
                if ANNOTATE_CN_ID:
                    axs[ax_idx].annotate(
                        f"{cn.sub_id}",
                        (x + width / 2, y + 0.25),
                        color="black",
                        weight="bold",
                        fontsize=10,
                        ha="center",
                        va="center",
                    )
        if layer_id not in layer_ids_seen:
            layer_ids_seen.append(layer_id)
            handles.append(handle)
            legend_labels.append(f"Layer {layer_id}")
    # print("Rectangles done...", end="")

    core_and_transfer_link_ids = sorted([core.id for core in accelerator.cores.node_list])
    hline_loc = core_and_transfer_link_ids[-1] - 0.5
    core_and_transfer_link_ids = core_and_transfer_link_ids[:-1]
    # Always define the last core as DRAM, not including DRAM in the core
    y_labels = [f"Core {core_id}" for core_id in core_and_transfer_link_ids]

    if plot_data_transfer:
        # First get all used and unique communication links
        used_cl_collect: list[CommunicationLink] = []
        for ky, pair_link in accelerator.communication_manager.pair_links.items():
            if pair_link:
                for link in pair_link:
                    if link.events and link not in used_cl_collect:
                        used_cl_collect.append(link)

        # Then plot the active data transfer period on these unique communication links
        pair_link_id = 0
        for cl in used_cl_collect:
            y_labels.append(cl.get_name_for_schedule_plot())

            """ Plot DRAM blocking period due to too_large_operand """
            for event in cl.events:
                task_type = event.type
                blocking = task_type.lower() == "block"
                start = event.start
                end = event.end
                runtime = end - start
                tensor = event.tensor
                weight_transfer = task_type.lower() == "transfer" and tensor.layer_operand in [
                    LayerOperand("W"),
                    LayerOperand("B"),
                ]
                layer_id = tensor.origin.id
                node_id = tensor.origin.sub_id
                if layer_id not in layer_ids_seen:
                    color = next(layer_colors)
                    colors_seen.append(color)
                else:
                    color = colors_seen[layer_ids_seen.index(layer_id)]
                x = start
                y = nb_cores - 1 + pair_link_id - 0.25
                width = runtime
                if blocking:
                    hatch = "xx"
                elif weight_transfer:
                    hatch = "---"
                else:
                    hatch = ""
                for ax_idx in range(0, nb_axs):
                    if (
                        (x_starts[ax_idx] <= x <= x_ends[ax_idx])
                        or (x_starts[ax_idx] <= x + width <= x_ends[ax_idx])
                        or (x_starts[ax_idx] > x and x + width > x_ends[ax_idx])
                    ):
                        axs[ax_idx].add_patch(
                            Rectangle(
                                xy=(x, y),
                                width=width,
                                height=height,
                                facecolor=color,
                                edgecolor="black",
                                lw=1,
                                hatch=hatch,
                                label=f"Layer {layer_id}",
                            )
                        )
                        if ANNOTATE_CN_ID:
                            axs[ax_idx].annotate(
                                f"{node_id}",
                                (x + width / 2, y + 0.25),
                                color="black",
                                weight="bold",
                                fontsize=10,
                                ha="center",
                                va="center",
                            )
            pair_link_id += 1

        """ Draw the divider line between schedule and data transfer """
        for ax in axs:
            ax.axhline(y=hline_loc, xmin=0, xmax=latency, c="black", linewidth=2, zorder=0)

    """ Plot inter-layer CN data dependency line """
    for prod, cons in G.edges():
        p_l = prod.id
        p_core = prod.chosen_core_allocation
        c_core = cons.chosen_core_allocation
        if not PLOT_DEPENDENCY_LINES_SAME_CORE and p_core == c_core:
            continue
        p_end = prod.end
        c_start = cons.start
        x1 = p_end
        x2 = c_start
        if draw_dependencies:
            for ax_idx in range(nb_axs):  # go through the different broken axes
                ax = axs[ax_idx]
                x_start = x_starts[ax_idx]
                x_end = x_ends[ax_idx]
                if (x_start <= x1 <= x_end) or (x_start <= x2 <= x_end):
                    line_start_x = p_end
                    if c_core < p_core:
                        line_start_y = p_core - 0.25
                    else:
                        line_start_y = p_core + 0.25
                    line_end_x = c_start
                    if c_core < p_core:
                        line_end_y = c_core + 0.25
                    else:
                        line_end_y = c_core - 0.25
                    color = colors_seen[layer_ids_seen.index(p_l)]
                    ax.plot(
                        [line_start_x, line_end_x],
                        [line_start_y, line_end_y],
                        color=color,
                        linewidth=dep_linewidth,
                    )

    bax.set_xlabel("Clock Cycles", labelpad=20, fontsize=14)

    plt.title(
        f"Latency = {int(latency):.3e} Cycles   Energy = {energy:.3e} pJ   EDP = {edp:.3e}",
        loc="right",
    )
    # Get all handles and labels and then filter them for unique ones and set legend
    legend_without_duplicate_labels(bax, loc=(0.0, 1.01), ncol=6)
    ylims = [ax.get_ylim() for ax in axs]
    miny = min((lim[0] for lim in ylims))
    maxy = max((lim[1] for lim in ylims))
    for ax in axs:
        ax.xaxis.set_major_formatter(major_formatter)
        ax.tick_params(axis="x", labelrotation=tick_rotation)
        ax.set_ylim((miny, maxy))
    bax.invert_yaxis()
    # replace all DRAM core (the core with the last core index) to 'DRAM'
    for i, label in enumerate(y_labels):
        y_labels[i] = label.replace(f"Core({accelerator.offchip_core_id})", "DRAM")
    axs[0].set_yticks(range(len(y_labels)))
    axs[0].set_yticklabels(y_labels)
    plt.show(block=False)
    plt.savefig(fig_path, format="png", bbox_inches="tight")
    logger.info(f"Plotted schedule timeline to {fig_path}")


def legend_without_duplicate_labels(bax, loc, ncol):
    handles, labels = zip(*bax.get_legend_handles_labels())
    handles = [item for sublist in handles for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
    unique.sort(key=lambda x: int(x[1].split(" ")[1]))  # Sort the labels based on the layer number
    bax.legend(*zip(*unique), loc=loc, ncol=ncol)


def major_formatter(x, pos):
    return f"{int(x):,}"


########################## PLOTLY PLOTTING ########################
def add_dependency_button(fig):
    show_bools = [True] * len(fig.data)
    hide_bools = [False if isinstance(trace, go.Scatter) else True for trace in fig.data]
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Hide dependencies",
                        "method": "update",
                        "args": [
                            {"visible": hide_bools},
                        ],
                    },
                    {
                        "label": "Show dependencies",
                        "method": "update",
                        "args": [
                            {"visible": show_bools},
                        ],
                    },
                ],
                "x": 0.99,
                "y": 1.01,
                "xanchor": "right",
                "yanchor": "bottom",
            }
        ]
    )


def add_dependencies(fig, scme, colors, layer_ids):
    for node in scme.workload.node_list:
        c_id = node.id
        c_l = node.id
        if c_l not in layer_ids:
            continue
        preds = scme.workload.predecessors(node)
        for pred in preds:
            p_id = pred.id
            p_l = pred.id
            if p_l == c_l:
                continue  # Ignore intra layer edges
            p_end = pred.end
            p_core = pred.chosen_core_allocation
            c_start = node.start
            c_core = node.chosen_core_allocation
            legendgroup = f"Layer {c_l}"
            legendgrouptitle_text = legendgroup
            marker = {"color": colors[c_l]}
            line = {"width": 1, "color": colors[c_l]}
            fig.add_trace(
                go.Scatter(
                    x=[p_end, c_start],
                    y=[f"Core {p_core}", f"Core {c_core}"],
                    name=f"{p_id}-->{c_id}",
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgrouptitle_text,
                    hoverinfo="none",
                    line=line,
                    marker=marker,
                )
            )
    # fig.for_each_trace(
    #     lambda trace: trace.update(visible=False) if "-->" in trace.name else (),
    # )


def get_communication_dicts(scme: "StreamCostModelEvaluation"):
    dicts = []
    accelerator: Accelerator = scme.accelerator

    active_links: set["CommunicationLink"] = set(
        link for link_pair in accelerator.communication_manager.pair_links.values() for link in link_pair if link.events
    )

    for cl in active_links:
        for cl_event in cl.events:
            task_type = cl_event.type
            start = cl_event.start
            end = cl_event.end
            runtime = end - start
            energy = cl_event.energy
            tensor = cl_event.tensor
            node = tensor.origin
            layer_id = node.id
            activity = cl_event.activity
            sender = cl_event.sender
            receiver = cl_event.receiver
            if runtime == 0:
                continue
            d = dict(
                Task=task_type.capitalize(),
                Id=np.nan,
                Sub_id=np.nan,
                Start=start,
                End=end,
                Resource=cl.get_name_for_schedule_plot(),
                Layer=layer_id,
                Runtime=runtime,
                Tensors={tensor: tensor.size},
                Type=task_type,
                Activity=activity,
                Energy=energy,
                LinkBandwidth=cl.bandwidth,
                Sender=sender,
                Receiver=receiver,
            )
            dicts.append(d)
    return dicts


def get_real_input_tensors(n, G):
    preds = list(G.predecessors(n))
    inputs = [pred.operand_tensors[pred.output_operand] for pred in preds if pred.id != n.id]
    inputs += [n.operand_tensors[op] for op in n.constant_operands]
    return inputs


def get_spatial_utilizations(
    scme: "StreamCostModelEvaluation", node: "ComputationNode", cost_lut: "CostModelEvaluationLUT"
):
    if cost_lut:
        equal_node = cost_lut.get_equal_node(node)
        assert (
            equal_node
        ), f"No equal node for {node} found in CostModelEvaluationLUT. Check LUT path (use the post-CO LUT when using CO)."
        core = scme.accelerator.get_core(node.chosen_core_allocation)
        cme = cost_lut.get_cme(equal_node, core)
        return cme.mac_spatial_utilization, cme.mac_utilization1
    return np.nan, np.nan


def get_energy_breakdown(
    scme: "StreamCostModelEvaluation", node: "ComputationNode", cost_lut: "CostModelEvaluationLUT"
):
    if cost_lut:
        equal_node = cost_lut.get_equal_node(node)
        assert (
            equal_node
        ), f"No equal node for {node} found in CostModelEvaluationLUT. Check LUT path (use the post-CO LUT when using CO)."
        core = scme.accelerator.get_core(node.chosen_core_allocation)
        cme = cost_lut.get_cme(equal_node, core)
        total_ops = cme.layer.total_mac_count
        en_total_per_op = cme.energy_total / total_ops
        en_breakdown = cme.mem_energy_breakdown
        en_breakdown_per_op = {}
        energy_sum_check = 0
        for layer_op, energies_for_all_levels in en_breakdown.items():
            d = {}
            mem_op = cme.layer.memory_operand_links[layer_op]
            for mem_level_idx, en in enumerate(energies_for_all_levels):
                mem_name = cme.accelerator.get_memory_level(mem_op, mem_level_idx).name
                d[mem_name] = en / total_ops
                energy_sum_check += en
            en_breakdown_per_op[layer_op] = d
        assert np.isclose(energy_sum_check, cme.mem_energy)
        return en_total_per_op, en_breakdown_per_op
    return np.nan, np.nan


def get_dataframe_from_scme(
    scme: "StreamCostModelEvaluation",
    layer_ids: list[int],
    add_communication: bool = False,
    cost_lut: "CostModelEvaluationLUT" = None,
):
    nodes = scme.workload.topological_sort()
    dicts = []
    for node in nodes:
        id = node.id
        layer = id
        if layer not in layer_ids:
            continue
        core_id = node.chosen_core_allocation
        start = node.start
        end = node.end
        runtime = node.runtime
        su_perfect_temporal, su_nonperfect_temporal = get_spatial_utilizations(scme, node, cost_lut)
        en_total_per_op, en_breakdown_per_op = get_energy_breakdown(scme, node, cost_lut)
        energy = node.onchip_energy
        tensors = get_real_input_tensors(node, scme.workload)
        task_type = "compute"
        d = dict(
            Task=node.short_name,
            Id=str(int(node.id)),
            Sub_id=str(int(node.sub_id)),
            Start=start,
            End=end,
            Resource=f"Core {core_id}",
            Layer=layer,
            Runtime=runtime,
            SpatialUtilization=su_perfect_temporal,
            SpatialUtilizationWithTemporal=su_nonperfect_temporal,
            Tensors={tensor: tensor.size for tensor in tensors},
            Type=task_type,
            Activity=np.nan,
            Energy=energy,
            EnergyTotalPerOp=en_total_per_op,
            EnergyBreakdownPerOp=en_breakdown_per_op,
        )
        dicts.append(d)
    if add_communication:
        communication_dicts = get_communication_dicts(scme)
        dicts += communication_dicts
    df = pd.DataFrame(dicts)
    return df


def get_sorted_y_labels(df):
    all_labels = set(df["Resource"].tolist())
    # Get computation labels
    computation_labels = [label for label in all_labels if "-" not in label]
    # Get communication labels (rest of the labels)
    communication_labels = [label for label in all_labels if label not in computation_labels]
    return sorted(computation_labels) + sorted(communication_labels)


def format_tensors(tensors: list[Tensor]):
    # Group tensors by their id attribute
    tensor_groups = defaultdict(list)
    for tensor in tensors:
        layer_id = tensor.id[0]
        tensor_groups[layer_id].append(tensor)

    # Format the tensor groups
    formatted_tensors = []
    for tensor_id, tensor_list in tensor_groups.items():
        if len(tensor_list) > 4:
            # Limit to 2 tensors at the beginning and 2 at the end
            formatted_tensors.append(f"{tensor_list[0]}, {tensor_list[1]}, ..., {tensor_list[-2]}, {tensor_list[-1]}")
        else:
            formatted_tensors.append(", ".join(map(str, tensor_list)))

    return "[<br>" + "<br>".join(formatted_tensors) + "<br>]"


def add_spatial_util_to_hovertext(hovertext: str, su_perfect_temporal: float, su_imperfect_temporal: float):
    if not isnan(su_perfect_temporal):
        hovertext += "<br><b>Spatial Utilization: </b><br>"
        hovertext += f"&nbsp;&nbsp;&nbsp;&nbsp;Without memory stalls: {su_perfect_temporal:.4f}<br>"
        hovertext += f"&nbsp;&nbsp;&nbsp;&nbsp;With memory stalls: {su_imperfect_temporal:.4f}<br>"
    return hovertext


def add_energy_breakdown_to_hovertext(
    hovertext: str, energy_total: float, energy_per_operation: float, energy_breakdown_per_op: dict
):
    if not isnan(energy_per_operation):
        hovertext += f"<b>Energy total: </b> {energy_total:.4e}<br>"
        hovertext += f"<b>Energy per operation:</b> {energy_per_operation:.4e}<br>"
        for layer_op, energy_dict in energy_breakdown_per_op.items():
            hovertext += f"<b>Energy breakdown for {layer_op}:</b><br>"
            for mem_level, en in energy_dict.items():
                hovertext += f"&nbsp;&nbsp;&nbsp;&nbsp;{mem_level}: {en:.4e}<br>"
    return hovertext


def add_activity_to_hovertext(hovertext: str, required_bandwidth: int, link_bandwidth: int):
    if not isnan(required_bandwidth) and not isnan(link_bandwidth):
        required_bandwidth = int(required_bandwidth)
        link_bandwidth = int(link_bandwidth)
        used_bandwidth = min(required_bandwidth, link_bandwidth)
        hovertext += f"<b>Required bandwidth:</b> {required_bandwidth} bits/cc<br>"
        hovertext += f"<b>Used bandwidth:</b> {used_bandwidth} bits/cc<br>"
    return hovertext


def visualize_timeline_plotly(
    scme: "StreamCostModelEvaluation",
    draw_dependencies: bool = False,
    draw_communication: bool = True,
    fig_path: str = "outputs/schedule.html",
    layer_ids: list[int] | None = None,
    cost_lut: CostModelEvaluationLUT = None,
):
    if not layer_ids:
        layer_ids = sorted(set(n.id for n in scme.workload.node_list))
    df = get_dataframe_from_scme(scme, layer_ids, draw_communication, cost_lut)
    # We get all the layer ids to get a color mapping for them
    layer_ids = sorted(list(set(df["Layer"].tolist())))
    color_cycle = cycle(sample_colorscale("rainbow", np.linspace(0, 1, len(layer_ids))))
    colors = {layer_id: c for (layer_id, c) in zip(layer_ids, color_cycle)}
    bars = []
    fig = go.Figure()
    seen_layers = []
    for idx, row in df.iterrows():
        id = row["Id"]
        sub_id = row["Sub_id"]
        start = row["Start"]
        runtime = row["Runtime"]
        su_perfect_temporal = row["SpatialUtilization"]
        su_imperfect_temporal = row["SpatialUtilizationWithTemporal"]
        energy = row["Energy"]
        energy_total_per_op = row["EnergyTotalPerOp"]
        energy_breakdown_per_op = row["EnergyBreakdownPerOp"]
        activity = row["Activity"]
        link_bandwidth = row["LinkBandwidth"]
        resource = row["Resource"]
        layer = row["Layer"]
        color = colors[layer]
        name = row["Task"]
        if isinstance((row["Id"]), str) and isinstance(row["Sub_id"], str):
            name += f"    <b>Id:</b> {id}    <b>Sub_id:</b> {sub_id}"
        legendgroup = f"Layer {layer}"
        legendgrouptitle_text = legendgroup
        tensors = format_tensors(row["Tensors"])
        task_type = row["Type"]
        hatch = PLOTLY_HATCH_TYPES[task_type]
        marker = {"color": color, "pattern": {"shape": hatch}}
        hovertext = (
            f"<b>Task:</b> {name}<br>"
            f"<b>Tensors:</b> {tensors}<br>"
            f"<b>Runtime:</b> {runtime:.2e}<br>"
            f"<b>Start:</b> {start:.4e}<br>"
            f"<b>End:</b> {start+runtime:.4e}<br>"
        )
        hovertext = add_activity_to_hovertext(hovertext, activity, link_bandwidth)
        hovertext = add_spatial_util_to_hovertext(hovertext, su_perfect_temporal, su_imperfect_temporal)
        hovertext = add_energy_breakdown_to_hovertext(hovertext, energy, energy_total_per_op, energy_breakdown_per_op)
        bar = go.Bar(
            base=[start],
            x=[runtime],
            y=[resource],
            name=name,
            orientation="h",
            marker=marker,
            legendgroup=legendgroup,
            legendgrouptitle_text=legendgrouptitle_text,
            hovertext=[hovertext],
            hoverinfo="text",
        )
        fig.add_trace(bar)
        bars.append(bar)
        seen_layers.append(layer)

    # Draw dependency lines if necessary
    if draw_dependencies:
        add_dependencies(fig, scme, colors, layer_ids)

    # Add button to show/hide dependencies
    add_dependency_button(fig)

    # Title
    edp = scme.latency * scme.energy
    fig.update_layout(
        title_text=(
            f"Computation Schedule.\t\t\tLatency = {scme.latency:.3e}\t\t\tEnergy = {scme.energy:.3e}\t\t\t"
            f"EDP = {edp:.3e}"
        )
    )
    # for bar in fig_timeline.data:
    #     fig.add_trace(go.Bar(bar), row=1,col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=get_sorted_y_labels(df))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="stack")
    fig.update_layout(showlegend=True)

    fig.write_html(fig_path)
    # fig.show()
    logger.info(f"Plotted schedule timeline using Plotly to {fig_path}.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    a = parser.add_argument("--path", "-p", type=str, help="Path to scme pickle file.")
    parser.add_argument(
        "--draw_dependencies",
        "-d",
        nargs="?",  # makes it optional
        default=0,
        type=int,
        help="Draw the inter-layer dependencies.",
    )
    parser.add_argument(
        "--draw_communication",
        "-c",
        nargs="?",
        default=0,
        type=int,
        help="Draw the inter-core communication.",
    )
    args = parser.parse_args()
    # Get scme from pickle filepath
    scme_path = args.path
    with open(scme_path, "rb") as fp:
        scme = pickle.load(fp)
    # Visualize using Plotly
    visualize_timeline_plotly(
        scme,
        draw_dependencies=args.draw_dependencies,
        draw_communication=args.draw_communication,
    )
