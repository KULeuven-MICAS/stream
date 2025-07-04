import logging
from typing import TYPE_CHECKING

import numpy as np
from brokenaxes import brokenaxes
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation

logger = logging.getLogger(__name__)

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


def humanbytes(bytes):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    bytes = float(bytes)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if bytes < KB:
        return f"{bytes} {'Bytes' if bytes == 0 or bytes > 1 else 'Byte'}"
    elif KB <= bytes < MB:
        return f"{bytes / KB:.2f} kB"
    elif MB <= bytes < GB:
        return f"{bytes / MB:.2f} MB"
    elif GB <= bytes < TB:
        return f"{bytes / GB:.2f} GB"
    elif TB <= bytes:
        return f"{bytes / TB:.2f} TB"
    else:
        raise ValueError(f"Cannot convert {bytes} bytes to human readable format.")


def plot_memory_usage(
    scme: "StreamCostModelEvaluation",
    section_start_percent: tuple[int, ...] = (0,),
    percent_shown: tuple[int, ...] = (100,),
    show_dram: bool = False,
    show_human_bytes: bool = True,
    fig_path: str = "outputs/memory_usage.png",
    fig_size: tuple[int, int] = (16, 9),
):
    memory_manager = scme.accelerator.memory_manager
    cpti, mopti = _filter_memory_instances(memory_manager, show_dram)
    cpti, mopti = _remove_zero_usage_instances(memory_manager, cpti, mopti)
    total_nb_of_top_instances = len(cpti)
    latency = _get_latency(scme)
    xlims = _calculate_xlims(latency, section_start_percent, percent_shown)
    fig, baxs = _create_figure_and_axes(total_nb_of_top_instances, xlims, fig_size)
    fig.suptitle("Memory usage through time (Bytes)")
    _plot_all_instances(baxs, cpti, mopti, memory_manager, xlims, show_human_bytes)
    fig.savefig(fig_path)
    logger.info(f"Saved memory usage fig to {fig_path}")


def _filter_memory_instances(memory_manager, show_dram):
    cpti = memory_manager.cores_per_top_instance.copy()
    mopti = memory_manager.memory_operands_per_top_instance.copy()
    if not show_dram:
        offchip_core_id = memory_manager.offchip_core_id
        to_remove = [
            top_instance
            for top_instance, cores in memory_manager.cores_per_top_instance.items()
            if all(core.id == offchip_core_id for core in cores)
        ]
        for top_instance in to_remove:
            cpti.pop(top_instance, None)
            mopti.pop(top_instance, None)
    return cpti, mopti


def _remove_zero_usage_instances(memory_manager, cpti, mopti):
    to_remove = []
    for ti, stored_cumsum in memory_manager.top_instance_stored_cumsum.items():
        _, stored_bits = zip(*stored_cumsum, strict=False)
        if max(stored_bits) == 0:
            to_remove.append(ti)
    for ti in to_remove:
        cpti.pop(ti, None)
        mopti.pop(ti, None)
    return cpti, mopti


def _get_latency(scme):
    assert scme.latency is not None, "SCME latency is None, cannot plot memory usage."
    return int(scme.latency)


def _calculate_xlims(latency, section_start_percent, percent_shown):
    x_starts = [int((start / 100) * latency) for start in section_start_percent]
    x_ends = [
        int(((start + percent) / 100) * latency)
        for (start, percent) in zip(section_start_percent, percent_shown, strict=False)
    ]
    return tuple(zip(x_starts, x_ends, strict=False))


def _create_figure_and_axes(total_nb_of_top_instances, xlims, fig_size):
    fig = plt.figure(figsize=fig_size)
    gridspecs = GridSpec(total_nb_of_top_instances, 1)
    baxs = [brokenaxes(xlims=xlims, subplot_spec=gridspec, wspace=0.05, d=0.005) for gridspec in gridspecs]
    return fig, baxs


def _plot_all_instances(baxs, cpti, mopti, memory_manager, xlims, show_human_bytes):
    for (
        ax,
        (ti_cores, cores_for_this_ti),
        (ti_memory_operands, memory_operands_for_this_ti),
    ) in zip(baxs, cpti.items(), mopti.items(), strict=False):
        assert ti_cores is ti_memory_operands, "Sanity check for same ordering of memory manager dicts failed."
        _plot_single_instance(
            ax,
            ti_cores,
            cores_for_this_ti,
            memory_operands_for_this_ti,
            memory_manager,
            xlims,
            show_human_bytes,
            is_last=(ax is baxs[-1]),
        )


def _plot_single_instance(
    ax,
    ti,
    cores_for_this_ti,
    memory_operands_for_this_ti,
    memory_manager,
    xlims,
    show_human_bytes,
    is_last,
):
    ti_cumsum = memory_manager.top_instance_stored_cumsum[ti]
    ti_cumsum_bytes = ti_cumsum.astype(float)
    ti_cumsum_bytes[:, 1] /= 8
    timesteps = ti_cumsum_bytes[:, 0]
    stored_bytes = ti_cumsum_bytes[:, 1]
    peak_usage_bytes = max(stored_bytes)
    if not peak_usage_bytes > 0:
        return
    if min(stored_bytes) < 0:
        logger.warn(f"We used negative amount of memory on top instance {ti}.")
    ax.plot(timesteps, stored_bytes, drawstyle="steps-post")
    ax.axhline(
        y=peak_usage_bytes,
        xmin=min(timesteps),
        xmax=max(timesteps),
        color="r",
        linestyle="dashed",
    )
    memory_capacity_bytes = memory_manager.top_instance_capacities[ti] / 8
    mem_text = (
        humanbytes(peak_usage_bytes) + " / " + humanbytes(np.array(memory_capacity_bytes))
        if show_human_bytes
        else f"{peak_usage_bytes} B /  {np.array(memory_capacity_bytes)} B"
    )
    ax.text(
        xlims[-1][1] - 1,
        peak_usage_bytes,
        mem_text,
        color="r",
        va="bottom",
        ha="right",
        fontsize=BIGGER_SIZE,
    )
    core_memory_operand_zipped = zip(cores_for_this_ti, memory_operands_for_this_ti, strict=False)
    formatted_str = "\n".join([f"{core}: {memory_operands}" for core, memory_operands in core_memory_operand_zipped])
    y_label = f"{ti.name}\n{formatted_str}"
    ax.set_ylim(bottom=0, top=1.05 * peak_usage_bytes)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if not is_last:
        ax.set_xticklabels([])
    ax.set_ylabel(
        y_label,
        rotation=0,
        fontsize=BIGGER_SIZE,
        labelpad=70,
        loc="center",
        va="center",
    )
