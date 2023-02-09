from matplotlib import pyplot as plt
import numpy as np

from stream.classes.cost_model.memory_manager import MemoryManager

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} kB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def plot_memory_usage(memory_manager: MemoryManager, show_dram = False, show_human_bytes = True, fig_path="outputs/memory_usage.png"):
    delta_history = memory_manager.delta_history
    total_nb_of_top_level_memories = sum([sum([len(history) > 1 for history in core_history if max(list(zip(*history))[1]) > 0]) for core_history in delta_history.values()])
    if show_dram == False:
        total_nb_of_top_level_memories=total_nb_of_top_level_memories-1;
    fig, axs = plt.subplots(total_nb_of_top_level_memories, figsize=(15, 6), sharex=True)
    fig.suptitle('Memory usage through time (Bytes)')
    axs_iter = iter(axs)
    cores_sorted = sorted([(core.id, core) for core in delta_history.keys()])

    if show_dram == False:
        cores_sorted.pop()

    for (core_id, core) in cores_sorted:
        core_history = delta_history[core]
        for top_level_idx, history in enumerate(core_history):
            if len(history) <= 1:
                continue  # Only plot memories that had changes throughout the scheduling
            memory_capacity = memory_manager.capacities[core][top_level_idx]/8
            # print(f"Max mem capacity for C{core_id} TL{top_level_idx} = {memory_capacity} \t")
            unique_timesteps = sorted(set((h[0] for h in history)))
            unique_timesteps_delta = [sum((delta for (timestep, delta) in history if timestep == unique_timestep)) for unique_timestep in unique_timesteps]
            history = sorted(history)  # sorted based on the timesteps
            # timesteps, used_deltas = zip(*history)
            timesteps = np.array(unique_timesteps)
            used_memory_space = np.cumsum(unique_timesteps_delta)/8
            # timesteps, used_memory_space = zip(*memory_manager.stored_cumsum[core][top_level_idx])
            assert min(used_memory_space) >= 0, f"We used negative amount of memory on core {core}."   
            if not max(used_memory_space) > 0:
                continue  # This happens for the weight memory on pooling core because it's encoded as zero bit precision
            ax = next(axs_iter)
            ax.plot(timesteps, used_memory_space, drawstyle='steps-post')  # Plot the timesteps and used memory through time
            ax.axhline(y=max(used_memory_space), xmin=min(timesteps), xmax=max(timesteps), color='r', linestyle='dashed')

            if show_human_bytes == True:
                mem_text = humanbytes(max(used_memory_space)) + ' / ' + humanbytes(np.array(memory_capacity))
            else:
                mem_text = f"{max(used_memory_space)} B /  {np.array(memory_capacity)} B"

            ax.text(min(timesteps), max(used_memory_space), mem_text, color='r', verticalalignment='bottom', fontsize=BIGGER_SIZE)
            ax.set_ylim(bottom=0, top=1.3 * max(used_memory_space))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            if show_dram and core_id == len(cores_sorted)-1:
                ax.set_ylabel(f"DRAM")
            else:
                if top_level_idx == 0:
                    ax.set_ylabel(f"Core-{core_id}\nWeight\n[Bytes]")
                else:
                    ax.set_ylabel(f"Core-{core_id}\nActivation\n[Bytes]")


    ax.set_xlabel("Cycles")  # Set xlabel of last axis (bottom one)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(fig_path)
    print(f"Saved memory usage fig to {fig_path}")



