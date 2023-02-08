from brokenaxes import brokenaxes
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from networkx import DiGraph
import numpy as np
import logging
import matplotlib.ticker as plticker

logger = logging.getLogger(__name__)

from stream.classes.hardware.architecture.accelerator import Accelerator


def plot_timeline_brokenaxes(G_given: DiGraph, accelerator: Accelerator, draw_dependencies: object = True,
                             section_start_percent: object = (0, 50, 95), percent_shown: object = (5, 5, 5),
                             plot_data_transfer: object = False,
                             fig_path: object = "outputs/schedule_plot.png") -> object:

    logger.info("Plotting...")

    nb_layers = len(set(iter([n.id[0] for n in G_given.nodes()])))
    nb_cores = accelerator.cores.number_of_nodes()

    dep_linewidth = 1
    ANNOTATE_CN_ID = True
    tick_rotation = 0
    assert len(section_start_percent) == len(percent_shown)

    # Copy the graph
    G = G_given.copy()

    # Total latency of the graph (including communication)
    # First get all used and unique communication links
    latency = max((cn.end for cn in G.nodes()))
    used_cl_collect = []
    for (ky, pair_link) in accelerator.pair_links.items():
        if pair_link:
            for link in pair_link:
                if (link.active_periods or link.blocked_periods) and link not in used_cl_collect:
                    used_cl_collect.append(link)
        # Then plot the active data transfer period on these unique communication links
        pair_link_id = 0
        for cl in used_cl_collect:
            try:
                latency = max(latency, cl.active_periods[-1][1])
            except:
                pass
    

    fig = plt.figure(figsize=(20, 6))

    x_starts = [int((start / 100) * latency) for start in section_start_percent]
    x_ends = [int(((start + percent) / 100) * latency * 1.01) for (start, percent) in zip(section_start_percent, percent_shown)]

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
    for cn in G.nodes():
        layer_id = cn.id[0]
        # Get the colour for this layer
        if layer_id not in layer_ids_seen:
            color = next(layer_colors)
            colors_seen.append(color)
        else:
            color = colors_seen[layer_ids_seen.index(layer_id)]
        x = cn.start# + 0.05
        y = cn.core_allocation - 0.25
        width = cn.runtime# - 0.05
        if (x_starts[0] <= x <= x_ends[0]) or (x_starts[0] <= x + width <= x_ends[0]) or (x_starts[0] > x and x + width > x_ends[0]):
            # First ax is done separately because of handle
            handle = axs[0].add_patch(Rectangle(xy=(x, y),
                                                width=width,
                                                height=height,
                                                facecolor=color,
                                                edgecolor='black',
                                                lw=1,
                                                label=f"Layer {layer_id}"))
            if ANNOTATE_CN_ID:
                axs[0].annotate(f"{cn.id[1]}", (x + width/2, y + 0.25), color='black', weight='bold', fontsize=10, ha='center', va='center')
        else:
            handle = None
        for ax_idx in range(1, nb_axs):
            if (x_starts[ax_idx] <= x <= x_ends[ax_idx]) or (x_starts[ax_idx] <= x + width <= x_ends[ax_idx]) or (x_starts[ax_idx] > x and x + width > x_ends[ax_idx]):
                axs[ax_idx].add_patch(Rectangle(xy=(x, y),
                                                width=width,
                                                height=height,
                                                facecolor=color,
                                                edgecolor='black',
                                                lw=1,
                                                label=f"Layer {layer_id}"))
                if ANNOTATE_CN_ID:
                    axs[ax_idx].annotate(f"{cn.id[1]}", (x + width/2, y + 0.25), color='black', weight='bold', fontsize=10, ha='center', va='center')
        if layer_id not in layer_ids_seen:
            layer_ids_seen.append(layer_id)
            handles.append(handle)
            legend_labels.append(f"Layer {layer_id}")
    # print("Rectangles done...", end="")

    core_and_transfer_link_ids = sorted([core.id for core in accelerator.cores.nodes()])
    hline_loc = core_and_transfer_link_ids[-1] - 0.5
    core_and_transfer_link_ids = core_and_transfer_link_ids[:-1]
    # Always define the last core as DRAM, not including DRAM in the core
    y_labels = [f"Core {core_id}" for core_id in core_and_transfer_link_ids]

    if plot_data_transfer:
        # First get all used and unique communication links
        used_cl_collect = []
        for (ky, pair_link) in accelerator.pair_links.items():
            if pair_link:
                for link in pair_link:
                    if (link.active_periods or link.blocked_periods) and link not in used_cl_collect:
                        used_cl_collect.append(link)

        # Then plot the active data transfer period on these unique communication links
        pair_link_id = 0
        for cl in used_cl_collect:
            y_labels.append(cl.get_name_for_schedule_plot())
            try:
                latency = max(latency, cl.active_periods[-1][1])
            except:
                pass

            """ Plot DRAM blocking period due to too_large_operand """
            if cl.blocked_periods:
                for blocked_period in cl.blocked_periods:
                    layer_id = blocked_period[2][0]
                    if layer_id not in layer_ids_seen:
                        color = next(layer_colors)
                        colors_seen.append(color)
                    else:
                        color = colors_seen[layer_ids_seen.index(layer_id)]
                    x = blocked_period[0]
                    y = nb_cores - 1 + pair_link_id - 0.25
                    width = blocked_period[1] - blocked_period[0]
                    for ax_idx in range(0, nb_axs):
                        if (x_starts[ax_idx] <= x <= x_ends[ax_idx]) or (x_starts[ax_idx] <= x + width <= x_ends[ax_idx]) or (x_starts[ax_idx] > x and x + width > x_ends[ax_idx]):
                            axs[ax_idx].add_patch(Rectangle(xy=(x, y),
                                                            width=width,
                                                            height=height,
                                                            facecolor=color,
                                                            edgecolor='black',
                                                            lw=1,
                                                            hatch='xx',
                                                            label=f"Layer {layer_id}"))
            """ Plot data transfer on communication link (active periods) """
            if cl.active_periods:
                for active_period in cl.active_periods:
                    layer_id = active_period[3][0]
                    # Get the colour for this layer
                    if layer_id not in layer_ids_seen:
                        color = next(layer_colors)
                        colors_seen.append(color)
                    else:
                        color = colors_seen[layer_ids_seen.index(layer_id)]
                    x = active_period[0]
                    y = nb_cores - 1 + pair_link_id - 0.25
                    width = active_period[1] - active_period[0]
                    # distinguish weight from activation (input/output)
                    if active_period[2] == 'W':
                        for ax_idx in range(0, nb_axs):
                            if (x_starts[ax_idx] <= x <= x_ends[ax_idx]) or (x_starts[ax_idx] <= x + width <= x_ends[ax_idx]) or (x_starts[ax_idx] > x and x + width > x_ends[ax_idx]):
                                axs[ax_idx].add_patch(Rectangle(xy=(x, y),
                                                                width=width,
                                                                height=height,
                                                                facecolor=color,
                                                                edgecolor='black',
                                                                lw=1,
                                                                hatch='---',
                                                                label=f"Layer {layer_id}"))
                    else:
                        for ax_idx in range(0, nb_axs):
                            if (x_starts[ax_idx] <= x <= x_ends[ax_idx]) or (x_starts[ax_idx] <= x + width <= x_ends[ax_idx]) or (x_starts[ax_idx] > x and x + width > x_ends[ax_idx]):
                                axs[ax_idx].add_patch(Rectangle(xy=(x, y),
                                                                width=width,
                                                                height=height,
                                                                facecolor=color,
                                                                edgecolor='black',
                                                                lw=1,
                                                                label=f"Layer {layer_id}"))
                                if ANNOTATE_CN_ID:
                                    axs[ax_idx].annotate(f"{active_period[3][1]}", (x + width/2, y + 0.25), color='black', weight='bold', fontsize=10, ha='center', va='center')
            pair_link_id += 1

        """ Draw the divider line between schedule and data transfer """
        for ax in axs:
            ax.axhline(y=hline_loc, xmin=0, xmax=latency, c="black", linewidth=2, zorder=0)

    """ Plot inter-layer CN data dependency line """
    for (prod, cons) in G.edges():
        p_l = prod.id[0]
        c_l = cons.id[0]
        p_core = prod.core_allocation
        c_core = cons.core_allocation
        if p_core == c_core:
            continue
        p_start = prod.start
        p_duration = prod.runtime
        p_end = prod.end
        c_start = cons.start
        c_duration = cons.runtime
        x1 = p_end
        y1 = p_core
        x2 = c_start
        y2 = c_core
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
                    ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], color=color, linewidth=dep_linewidth)

    bax.set_xlabel('Clock Cycles', labelpad=20, fontsize=14)

    plt.title(f"Latency = {int(latency):,} Cycles", loc='right')
    # Get all handles and labels and then filter them for unique ones and set legend
    legend_without_duplicate_labels(bax, loc=(0.0, 1.01), ncol=9)
    ylims = [ax.get_ylim() for ax in axs]
    miny = min((lim[0] for lim in ylims))
    maxy = max((lim[1] for lim in ylims))
    for ax in axs:
        ax.xaxis.set_major_formatter(major_formatter)
        ax.tick_params(axis='x', labelrotation=tick_rotation)
        ax.set_ylim((miny, maxy))
    bax.invert_yaxis()
    # replace all DRAM core (the core with the last core index) to 'DRAM'
    for i, label in enumerate(y_labels):
        y_labels[i] = label.replace(f'Core({accelerator.offchip_core_id})', 'DRAM')
    axs[0].set_yticks(range(len(y_labels)))
    axs[0].set_yticklabels(y_labels)
    # plt.show()
    plt.savefig(fig_path, format="png", bbox_inches='tight')
    print(f"Saved timeline fig to {fig_path}")


def legend_without_duplicate_labels(bax, loc, ncol):
    handles, labels = zip(*bax.get_legend_handles_labels())
    handles = [item for sublist in handles for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    unique.sort(key=lambda x:int(x[1].split(" ")[1]))  # Sort the labels based on the layer number
    bax.legend(*zip(*unique), loc=loc, ncol=ncol)


def major_formatter(x, pos):
    return f'{int(x):,}'
