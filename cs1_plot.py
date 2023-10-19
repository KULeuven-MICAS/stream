import glob
import matplotlib.pyplot as plt
import numpy as np

from stream.utils import load_scme

HW_SINGLE_CORE = ["single_core_16x16_mesh_dpDRAM", "single_core_32x32_mesh_dpDRAM",
              "single_core_64x64_mesh_dpDRAM", "single_core_128x128_mesh_dpDRAM"]
HW_HOMOGENEOUS_MULTI_CORE = ["single_core_16x16_mesh_dpDRAM", "HW2_4homo_mesh_dpDRAM",
                "HW500_16homo_mesh_dpDRAM", "HW700_64homo_mesh_dpDRAM"]
HW_HETEROGENEOUS_MULTI_CORE = ["single_core_16x16_mesh_dpDRAM", "HW3_4hetero_mesh_dpDRAM",
                  "HW600_16hetero_mesh_dpDRAM", "HW800_64hetero_mesh_dpDRAM"]
SCHEDULING_TYPES = ["lbl", "fused"]

WORKLOADS = ['resnet18', 'fsrcnn', 'mobilenetv2', 'squeezenet', 'inception_v2']
# WORKLOADS = ['resnet18']


def plot_cs1(opt_goal, wl, scme_path, plot_path):
    sc_result_collect = {'fused':[], 'lbl':[]}
    homo_result_collect = {'fused':[], 'lbl':[]}
    hetero_result_collect = {'fused':[], 'lbl':[]}
    for i, hw in enumerate(HW_SINGLE_CORE + HW_HOMOGENEOUS_MULTI_CORE + HW_HETEROGENEOUS_MULTI_CORE):
        for st in SCHEDULING_TYPES:
            ky_id = f"{hw}-{wl}-hintloop_*-{st}"
            path = glob.glob(f'{scme_path}*{ky_id}*')
            if len(path) == 0:
                raise ValueError(f"SCME for {ky_id} not found. Might have crashed or is not finished yet.")
            elif len(path) > 1:
                raise ValueError(f"More than one path matching pattern. Pattern not correct?")
            scme = load_scme(path[0])
            en = scme.energy
            lat = scme.latency
            edp = en * lat
            if i < len(HW_SINGLE_CORE):  # can't use in operator because some hw are in multiple lists
                sc_result_collect[st].append((lat, en, edp))
            elif len(HW_SINGLE_CORE) <= i < len(HW_SINGLE_CORE) + len(HW_HOMOGENEOUS_MULTI_CORE):
                homo_result_collect[st].append((lat, en, edp))
            else:
                hetero_result_collect[st].append((lat, en, edp))



    sc_color = 'red'
    homo_color = 'blue'
    hetero_color = 'green'

    title = ['Latency', 'Energy', 'EDP']
    unit = ['cc', 'pJ', 'cc*pJ']
    plt.figure(figsize=(16,6))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        sc_lbl = np.array([x[i] for x in sc_result_collect['lbl']])
        plt.plot(sc_lbl, color=sc_color, marker='o', label='single-core lbl')
        sc_fused = np.array([x[i] for x in sc_result_collect['fused']])
        plt.plot(sc_fused, linestyle='dotted', color=sc_color, marker='o', label='single-core fused')

        homo_lbl = np.array([x[i] for x in homo_result_collect['lbl']])
        plt.plot(homo_lbl, color=homo_color, marker='o', label='homo lbl')
        homo_fused = np.array([x[i] for x in homo_result_collect['fused']])
        plt.plot(homo_fused, linestyle='dotted', color=homo_color, marker='o', label='homo fused')

        hetero_lbl = np.array([x[i] for x in hetero_result_collect['lbl']])
        plt.plot(hetero_lbl, color=hetero_color, marker='o', label='hetero lbl')
        hetero_fused = np.array([x[i] for x in hetero_result_collect['fused']])
        plt.plot(hetero_fused, linestyle='dotted', color=hetero_color, marker='o', label='hetero fused')

        if i == 0:
            plt.legend()
        plt.ylabel(f"{title[i]} ({unit[i]})")
        plt.xticks([0,1,2,3], ['16x16\n1 core', '32x32\n4 cores', '64x64\n16 cores', '128x128\n64 cores'])
        plt.xlabel("Computation Array Size")
        plt.title(f"{title[i]} ({wl} - opt. for {opt_goal})")

    plt.tight_layout()
    plot_file_path = f"{plot_path}{wl}_{opt_goal}.png"
    plt.savefig(plot_file_path)
    print(f"Plotted: {plot_file_path}")

if __name__ == "__main__":
    for opt_goal in ['latency']:
        scme_path = f'/users/micas/asymons/stream_TC_2023/exploration_k_split_memory_energy_coarser_4/inter_result/'
        plot_path = f'/users/micas/asymons/stream_TC_2023/exploration_k_split_memory_energy_coarser_4/cs1_plot/'
        for wl in WORKLOADS:
            try:
               plot_cs1(opt_goal, wl, scme_path, plot_path)
            except:
                print(f"WARNING: {wl} plot failed.")
                continue