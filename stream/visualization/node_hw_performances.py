import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Run these if this doesn't work:
# $ sudo apt install msttcorefonts -qq
# $ rm ~/.cache/matplotlib -rf
matplotlib.rc("font", family="Arial")

# MPL FONT SIZES
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 20


def autolabel(rects, ax, indices=[], labels=[], offsets=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    index = 0
    if offsets is None:
        offsets = [0 for _ in range(len(labels))]
    offsets = iter(offsets)
    for rect in rects:
        plt.rcParams.update({"font.size": MEDIUM_SIZE})
        # height = rect.get_height()
        # ax.annotate('{}'.format(height/1e6),
        #             xy=(rect.get_x() + rect.get_width() / 2, height-0.4e6),
        #             xytext=(0, 3),  # 3 points vertical offset
        #             textcoords="offset points",
        #             ha='center', va='bottom')

        if index in indices:
            plt.rcParams.update({"font.size": MEDIUM_SIZE})
            height = rect.get_height()
            ax.annotate(
                labels.pop(0),
                xy=(rect.get_x() + rect.get_width() / 2 + 0.03, height + next(offsets)),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                weight="bold",
            )
        else:
            raise ValueError(f"Index {index} not in indices.")
        index += 1


def visualize_node_hw_performances_pickle(
    pickle_filepath, scale_factors=None, fig_path=None
):
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    with open(pickle_filepath, "rb") as handle:
        node_hw_performances = pickle.load(handle)

    # first_key = next(iter(node_hw_performances))
    # node_hw_performances = {first_key: node_hw_performances[first_key]}

    if not scale_factors:
        scale_factors = {node: 1 for node in node_hw_performances}
    else:
        scale_factors = {
            next(node for node in node_hw_performances if node.id == k): v
            for k, v in scale_factors.items()
        }

    if not fig_path:
        basename = os.path.basename(pickle_filepath).split(".")[0]
        fig_path = f"outputs/{basename}.png"

    node_labels = []
    cores = []
    min_latency_per_node = {}
    min_energy_per_node = {}
    for node, hw_performances in node_hw_performances.items():
        node_labels.append(f"L{node.id[0]}\nN{node.id[1]}")
        min_latency_per_node[node] = float("inf")
        min_energy_per_node[node] = float("inf")
        for core, cme in hw_performances.items():
            if core not in cores:
                cores.append(core)
            if cme.latency_total2 < min_latency_per_node[node]:
                min_latency_per_node[node] = cme.latency_total2
                min_energy_per_node[node] = cme.energy_total
        # print(
        #     node.type,
        #     node.id,
        #     {k: cme.latency_total2 for k, cme in hw_performances.items()},
        # )
    # Multiply the min_latency_per_node and min_energy_per_node with the scale factor
    for node in min_latency_per_node:
        min_latency_per_node[node] *= scale_factors[node]
        min_energy_per_node[node] *= scale_factors[node]
    # Sum the latencies (assumes no overlap)
    worst_case_latency = sum(min_latency_per_node.values())
    # Sum the energies (assumes every node is mapped to most energy-efficient core)
    best_case_energy = sum(min_energy_per_node.values())

    # COLORMAP
    colormap = list(plt.cm.rainbow(np.linspace(0, 1, len(cores))))
    # colormap = plt.get_cmap("Set1")
    colors = {core: colormap[i] for i, core in enumerate(cores)}

    x = np.arange(len(node_labels))
    width = 0.8 / len(cores)
    offsets = [
        -(width / 2) - ((len(cores) - 1) // 2) * width + i * width
        for i in range(len(cores))
    ]

    fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    for core, offset in zip(cores, offsets):
        core_latencies = []
        core_energies = []
        for node, hw_performances in node_hw_performances.items():
            if core in hw_performances:
                core_latencies.append(
                    scale_factors[node] * hw_performances[core].latency_total2
                )
                core_energies.append(
                    scale_factors[node] * hw_performances[core].energy_total
                )
            else:
                core_latencies.append(0)
                core_energies.append(0)
        for ax, y_values in zip(axs, [core_latencies, core_energies]):
            rects = ax.bar(
                x + offset, y_values, width, label=f"{core}", color=colors[core]
            )
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(node_labels)
        ax.set_yscale("log")
        ax.yaxis.grid(which="major", linestyle="-", linewidth=0.5, color="black")
        ax.yaxis.grid(
            which="minor", linestyle=":", linewidth=0.25, color=(0.2, 0.2, 0.2)
        )
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=min(len(cores), 7))
    axs[0].set_title(
        f"Worst (no overlap) best-case latency = {worst_case_latency:.3e} Cycles",
        loc="right",
    )
    axs[0].set_ylabel("Latency [cycles]")
    axs[1].set_title(
        f"Best-case energy = {best_case_energy:.3e} pJ",
        loc="right",
    )
    axs[1].set_ylabel("Energy [pJ]")
    fig.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    logger.info(f"Saved node_hw_performances visualization to: {fig_path}")
