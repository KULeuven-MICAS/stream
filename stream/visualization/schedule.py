import logging
from typing import TYPE_CHECKING

import numpy as np
from brokenaxes import brokenaxes
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from zigzag.datatypes import LayerOperand

from stream.hardware.architecture.noc.communication_link import CommunicationLink

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.onnx_workload import ComputationNodeWorkload

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
    G: ComputationNodeWorkload = scme.workload
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
    assert isinstance(latency, int), f"Latency should be an integer, got {type(latency)}."
    # Total energy of the SCME
    energy = scme.energy
    assert isinstance(energy, float), f"Energy should be a float, got {type(energy)}."
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
    layer_colors = iter(plt.cm.rainbow(np.linspace(0, 1, nb_layers)))  # type: ignore
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
        assert cn.chosen_core_allocation is not None, f"Chosen core allocation for {cn} is None."
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
        for paths in accelerator.communication_manager.all_pair_links.values():
            for pair_link in paths:
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
