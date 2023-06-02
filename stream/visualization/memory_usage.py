from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from brokenaxes import brokenaxes

from stream.classes.cost_model.memory_manager import MemoryManager

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if B < KB:
        return "{0} {1}".format(B, "Bytes" if 0 == B > 1 else "Byte")
    elif KB <= B < MB:
        return "{0:.2f} kB".format(B / KB)
    elif MB <= B < GB:
        return "{0:.2f} MB".format(B / MB)
    elif GB <= B < TB:
        return "{0:.2f} GB".format(B / GB)
    elif TB <= B:
        return "{0:.2f} TB".format(B / TB)


def plot_memory_usage(
    scme,
    section_start_percent=(0,),
    percent_shown=(100,),
    show_dram=False,
    show_human_bytes=True,
    fig_path="outputs/memory_usage.png",
    fig_size=(16, 9),
):
    memory_manager = scme.accelerator.memory_manager
    cpti = memory_manager.cores_per_top_instance.copy()
    mopti = memory_manager.memory_operands_per_top_instance.copy()

    if not show_dram:
        # Remove the dram memory instance(s)
        offchip_core_id = memory_manager.offchip_core_id
        for top_instance, cores in memory_manager.cores_per_top_instance.items():
            if all([core.id == offchip_core_id for core in cores]):
                del cpti[top_instance]
                del mopti[top_instance]

    # Get the max timesteps across all top instances to position the text
    # Moreover, if there's an instance that doesn't have a stored cumsum larger than 0,
    # don't consider it.
    max_timestep_across_all_tis = 0
    for ti, stored_cumsum in memory_manager.top_instance_stored_cumsum.items():
        timesteps, stored_bits = zip(*stored_cumsum)
        if max(timesteps) > max_timestep_across_all_tis:
            max_timestep_across_all_tis = max(timesteps)
        if max(stored_bits) == 0:
            del cpti[ti]
            del mopti[ti]
    # Total number of subplots we will draw
    total_nb_of_top_instances = len(cpti)

    # Total latency of the SCME
    latency = scme.latency
    # Calculate the brokenaxes x ranges based on the given start and show percentage
    x_starts = [int((start / 100) * latency) for start in section_start_percent]
    x_ends = [
        int(((start + percent) / 100) * latency)
        for (start, percent) in zip(section_start_percent, percent_shown)
    ]
    xlims = tuple(zip(x_starts, x_ends))

    fig = plt.figure(figsize=fig_size)
    gridspecs = GridSpec(total_nb_of_top_instances, 1)
    baxs = [
        brokenaxes(xlims=xlims, subplot_spec=gridspec, wspace=0.05, d=0.005)
        for gridspec in gridspecs
    ]

    fig.suptitle("Memory usage through time (Bytes)")

    peak_usages_bytes = {}
    for (
        ax,
        (ti_cores, cores_for_this_ti),
        (ti_memory_operands, memory_operands_for_this_ti),
    ) in zip(baxs, cpti.items(), mopti.items()):
        assert (
            ti_cores is ti_memory_operands
        ), "Sanity check for same ordering of memory manager dicts failed."
        ti = ti_cores
        ti_cumsum = memory_manager.top_instance_stored_cumsum[ti]
        ti_cumsum_bytes = ti_cumsum.astype(float)
        ti_cumsum_bytes[:, 1] /= 8
        timesteps = ti_cumsum_bytes[:, 0]
        stored_bytes = ti_cumsum_bytes[:, 1]
        peak_usage_bytes = max(stored_bytes)
        peak_usages_bytes[ti] = peak_usage_bytes
        if not peak_usage_bytes > 0:
            continue  # Happens for weight memory on pooling core because it's encoded as zero bit
        assert (
            min(stored_bytes) >= 0
        ), f"We used negative amount of memory on top instance {ti}."
        ax.plot(
            timesteps, stored_bytes, drawstyle="steps-post"
        )  # Plot the timesteps and used memory through time
        ax.axhline(
            y=peak_usage_bytes,
            xmin=min(timesteps),
            xmax=max(timesteps),
            color="r",
            linestyle="dashed",
        )
        memory_capacity_bytes = memory_manager.top_instance_capacities[ti] / 8
        if show_human_bytes:
            mem_text = (
                humanbytes(peak_usage_bytes)
                + " / "
                + humanbytes(np.array(memory_capacity_bytes))
            )
        else:
            mem_text = f"{peak_usage_bytes} B /  {np.array(memory_capacity_bytes)} B"
        ax.text(
            xlims[-1][1] - 1,
            peak_usage_bytes,
            mem_text,
            color="r",
            va="bottom",
            ha="right",
            fontsize=BIGGER_SIZE,
        )
        core_memory_operand_zipped = zip(cores_for_this_ti, memory_operands_for_this_ti)
        formatted_str = "\n".join(
            [
                f"{core}: {memory_operands}"
                for core, memory_operands in core_memory_operand_zipped
            ]
        )
        y_label = f"{ti.name}\n{formatted_str}"
        # ax.set_xlim(left=0, right=max(timesteps))
        ax.set_ylim(bottom=0, top=1.05 * peak_usage_bytes)

        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if ax is not baxs[-1]:
            ax.set_xticklabels([])
        ax.set_ylabel(
            y_label,
            rotation=0,
            fontsize=BIGGER_SIZE,
            labelpad=70,
            loc="center",
            va="center",
        )

    # ax.set_xlabel("Cycles")  # Set xlabel of last axis (bottom one)
    # plt.show(block=True)
    fig.savefig(fig_path)
    print(f"Saved memory usage fig to {fig_path}")
