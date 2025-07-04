import logging
from typing import TYPE_CHECKING, Any

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
from dataclasses import dataclass

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


def _set_matplotlib_rc(nb_layers: int) -> None:
    MAX_NB_LAYERS_FOR_MEDIUM_FONT = 6
    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    plt.rc("axes", labelsize=BIGGER_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALLER_SIZE if nb_layers > MAX_NB_LAYERS_FOR_MEDIUM_FONT else MEDIUM_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)


def _get_latency_energy_edp(scme):
    latency = scme.latency
    assert isinstance(latency, int), f"Latency should be an integer, got {type(latency)}."
    energy = scme.energy
    assert isinstance(energy, float), f"Energy should be a float, got {type(energy)}."
    edp = latency * energy
    return latency, energy, edp


def _create_broken_axes(latency, section_start_percent, percent_shown):
    assert len(section_start_percent) == len(percent_shown)
    x_starts = [int((start / 100) * latency) for start in section_start_percent]
    x_ends = [
        int(((start + percent) / 100) * latency)
        for (start, percent) in zip(section_start_percent, percent_shown, strict=False)
    ]
    bax = brokenaxes(xlims=tuple(zip(x_starts, x_ends, strict=False)), wspace=0.05, d=0.005)
    axs = bax.axs
    return x_starts, x_ends, bax, axs


def _init_layer_colors(nb_layers):
    layer_colors = iter(plt.cm.rainbow(np.linspace(0, 1, nb_layers)))  # type: ignore
    colors_seen = []
    handles = []
    layer_ids_seen = []
    legend_labels = []
    return layer_colors, colors_seen, handles, layer_ids_seen, legend_labels


def _get_layer_color(layer_id, layer_ids_seen, layer_colors, colors_seen):
    if layer_id not in layer_ids_seen:
        color = next(layer_colors)
        colors_seen.append(color)
    else:
        color = colors_seen[layer_ids_seen.index(layer_id)]
    return color


def _draw_divider_line(axs, hline_loc, latency):
    for ax in axs:
        ax.axhline(y=hline_loc, xmin=0, xmax=latency, c="black", linewidth=2, zorder=0)


def _finalize_plot(bax, axs, y_labels, accelerator, latency, energy, edp, fig_path):
    bax.set_xlabel("Clock Cycles", labelpad=20, fontsize=14)
    plt.title(
        f"Latency = {int(latency):.3e} Cycles   Energy = {energy:.3e} pJ   EDP = {edp:.3e}",
        loc="right",
    )
    legend_without_duplicate_labels(bax, loc=(0.0, 1.01), ncol=6)
    ylims = [ax.get_ylim() for ax in axs]
    miny = min(lim[0] for lim in ylims)
    maxy = max(lim[1] for lim in ylims)
    for ax in axs:
        ax.xaxis.set_major_formatter(major_formatter)
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_ylim((miny, maxy))
    bax.invert_yaxis()
    for i, label in enumerate(y_labels):
        y_labels[i] = label.replace(f"Core({accelerator.offchip_core_id})", "DRAM")
    axs[0].set_yticks(range(len(y_labels)))
    axs[0].set_yticklabels(y_labels)
    plt.show(block=False)
    plt.savefig(fig_path, format="png", bbox_inches="tight")
    logger.info(f"Plotted schedule timeline to {fig_path}")


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

    @dataclass
    class PlotContext:
        G: "ComputationNodeWorkload"
        accelerator: "Accelerator"
        nb_layers: int
        nb_cores: int
        layer_colors: Any
        colors_seen: list
        handles: list
        layer_ids_seen: list
        legend_labels: list
        x_starts: list
        x_ends: list
        axs: list
        nb_axs: int

    nb_layers = len(set(n.id for n in G.node_list))
    nb_cores = accelerator.cores.number_of_nodes()

    _set_matplotlib_rc(nb_layers)
    logger.info("Plotting...")

    latency, energy, edp = _get_latency_energy_edp(scme)
    plt.figure(figsize=(20, 6))

    x_starts, x_ends, bax, axs = _create_broken_axes(latency, section_start_percent, percent_shown)
    nb_axs = len(axs)

    layer_colors, colors_seen, handles, layer_ids_seen, legend_labels = _init_layer_colors(nb_layers)

    ctx = PlotContext(
        G=G,
        accelerator=accelerator,
        nb_layers=nb_layers,
        nb_cores=nb_cores,
        layer_colors=layer_colors,
        colors_seen=colors_seen,
        handles=handles,
        layer_ids_seen=layer_ids_seen,
        legend_labels=legend_labels,
        x_starts=x_starts,
        x_ends=x_ends,
        axs=axs,
        nb_axs=nb_axs,
    )

    y_labels, hline_loc = _plot_computation_nodes(ctx)

    if plot_data_transfer:
        y_labels = _plot_data_transfers(ctx, y_labels)
        _draw_divider_line(ctx.axs, hline_loc, latency)

    if draw_dependencies:
        _plot_dependency_lines(ctx)

    _finalize_plot(bax, ctx.axs, y_labels, accelerator, latency, energy, edp, fig_path)


def _plot_computation_nodes(ctx):
    height = 0.5
    core_and_transfer_link_ids = sorted([core.id for core in ctx.G.accelerator.cores.node_list])
    hline_loc = core_and_transfer_link_ids[-1] - 0.5
    core_and_transfer_link_ids = core_and_transfer_link_ids[:-1]
    y_labels = [f"Core {core_id}" for core_id in core_and_transfer_link_ids]

    for cn in ctx.G.node_list:
        layer_id = cn.id
        color = _get_layer_color(layer_id, ctx.layer_ids_seen, ctx.layer_colors, ctx.colors_seen)
        x = cn.start
        assert cn.chosen_core_allocation is not None, f"Chosen core allocation for {cn} is None."
        y = cn.chosen_core_allocation - 0.25
        width = cn.runtime

        handle = _add_rectangle_to_axes(ctx, x, y, width, height, color, layer_id, cn.sub_id)
        if layer_id not in ctx.layer_ids_seen:
            ctx.layer_ids_seen.append(layer_id)
            ctx.handles.append(handle)
            ctx.legend_labels.append(f"Layer {layer_id}")
    return y_labels, hline_loc


def _add_rectangle_to_axes(ctx, x, y, width, height, color, layer_id, sub_id):
    ANNOTATE_CN_ID = True
    handle = None
    for ax_idx in range(ctx.nb_axs):
        if (
            (ctx.x_starts[ax_idx] <= x <= ctx.x_ends[ax_idx])
            or (ctx.x_starts[ax_idx] <= x + width <= ctx.x_ends[ax_idx])
            or (ctx.x_starts[ax_idx] > x and x + width > ctx.x_ends[ax_idx])
        ):
            rect = Rectangle(
                xy=(x, y),
                width=width,
                height=height,
                facecolor=color,
                edgecolor="black",
                lw=1,
                label=f"Layer {layer_id}",
            )
            h = ctx.axs[ax_idx].add_patch(rect)
            if ax_idx == 0:
                handle = h
            if ANNOTATE_CN_ID:
                ctx.axs[ax_idx].annotate(
                    f"{sub_id}",
                    (x + width / 2, y + 0.25),
                    color="black",
                    weight="bold",
                    fontsize=10,
                    ha="center",
                    va="center",
                )
    return handle


def _plot_data_transfers(ctx, y_labels):
    used_cl_collect: list[CommunicationLink] = []
    for paths in ctx.accelerator.communication_manager.all_pair_links.values():
        for pair_link in paths:
            if pair_link:
                for link in pair_link:
                    if link.events and link not in used_cl_collect:
                        used_cl_collect.append(link)

    pair_link_id = 0
    height = 0.5
    for cl in used_cl_collect:
        y_labels.append(cl.get_name_for_schedule_plot())
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
            color = _get_layer_color(layer_id, ctx.layer_ids_seen, ctx.layer_colors, ctx.colors_seen)
            x = start
            y = ctx.nb_cores - 1 + pair_link_id - 0.25
            width = runtime
            hatch = "xx" if blocking else ("---" if weight_transfer else "")
            _add_transfer_rectangle_to_axes(ctx, x, y, width, height, color, hatch, layer_id, node_id)
        pair_link_id += 1
    return y_labels


def _add_transfer_rectangle_to_axes(ctx, x, y, width, height, color, hatch, layer_id, node_id):  # noqa: PLR0913
    ANNOTATE_CN_ID = True
    for ax_idx in range(ctx.nb_axs):
        if (
            (ctx.x_starts[ax_idx] <= x <= ctx.x_ends[ax_idx])
            or (ctx.x_starts[ax_idx] <= x + width <= ctx.x_ends[ax_idx])
            or (ctx.x_starts[ax_idx] > x and x + width > ctx.x_ends[ax_idx])
        ):
            ctx.axs[ax_idx].add_patch(
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
                ctx.axs[ax_idx].annotate(
                    f"{node_id}",
                    (x + width / 2, y + 0.25),
                    color="black",
                    weight="bold",
                    fontsize=10,
                    ha="center",
                    va="center",
                )


def _plot_dependency_lines(ctx):
    dep_linewidth = 1
    for prod, cons in ctx.G.edges():
        p_l = prod.id
        p_core = prod.chosen_core_allocation
        c_core = cons.chosen_core_allocation
        if not PLOT_DEPENDENCY_LINES_SAME_CORE and p_core == c_core:
            continue
        p_end = prod.end
        c_start = cons.start
        x1 = p_end
        x2 = c_start
        for ax_idx in range(ctx.nb_axs):
            ax = ctx.axs[ax_idx]
            x_start = ctx.x_starts[ax_idx]
            x_end = ctx.x_ends[ax_idx]
            if (x_start <= x1 <= x_end) or (x_start <= x2 <= x_end):
                line_start_x = p_end
                line_start_y = p_core - 0.25 if c_core < p_core else p_core + 0.25
                line_end_x = c_start
                line_end_y = c_core + 0.25 if c_core < p_core else c_core - 0.25
                color = ctx.colors_seen[ctx.layer_ids_seen.index(p_l)]
                ax.plot(
                    [line_start_x, line_end_x],
                    [line_start_y, line_end_y],
                    color=color,
                    linewidth=dep_linewidth,
                )


def legend_without_duplicate_labels(bax, loc, ncol):
    handles, labels = zip(*bax.get_legend_handles_labels(), strict=False)
    handles = [item for sublist in handles for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    unique = [
        (handle, label)
        for i, (handle, label) in enumerate(zip(handles, labels, strict=False))
        if label not in labels[:i]
    ]
    unique.sort(key=lambda x: int(x[1].split(" ")[1]))  # Sort the labels based on the layer number
    bax.legend(*zip(*unique, strict=False), loc=loc, ncol=ncol)


def major_formatter(x, pos):
    return f"{int(x):,}"
