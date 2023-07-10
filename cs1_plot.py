import glob
import re
import pickle
import pandas as pd
import xlsxwriter

HW_sc_list = ["single_core_16x16_mesh_dpDRAM", "single_core_32x32_mesh_dpDRAM",
              "single_core_64x64_mesh_dpDRAM", "single_core_128x128_mesh_dpDRAM"]
HW_homo_list = ["single_core_16x16_mesh_dpDRAM", "HW2_4homo_mesh_dpDRAM",
                "HW500_16homo_mesh_dpDRAM", "HW700_64homo_mesh_dpDRAM"]
HW_hetero_list = ["single_core_16x16_mesh_dpDRAM", "HW3_4hetero_mesh_dpDRAM",
                  "HW600_16hetero_mesh_dpDRAM", "HW800_64hetero_mesh_dpDRAM"]
schedule_way = ["lyl", "fused"]

WL_list = ['resnet18', 'fsrcnn', 'mobilenetv2', 'squeezenet', 'inception_v2']
# WL_list = ['resnet18']


def plot_cs1(opt_goal, WL, result_path):
    WL_list = [WL]
    sc_result_collect = {'fused':[], 'lyl':[]}
    for hw in HW_sc_list:
        for wl in WL_list:
            for sw in schedule_way:
                ky_id = f"{sw}_{hw}_{wl}"
                path = glob.glob(f'{result_path}/*{ky_id}.log')
                assert len(path) == 1, f"Two results found for one design point {ky_id}, something is wrong."
                print(f'Reading in result -- {path}')
                en = None
                la = None
                edp = None
                with open(path[0], 'rb') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        mark = str(line).split()
                        if len(mark) > 20:
                            if mark[3] == "__main__.<module>":
                                en = float(mark[17])
                                la = float(mark[13])
                                edp = float(mark[21][:-3])
                                break
                    if la is None:
                        print(f"{ky_id} not done...")
                    sc_result_collect[sw].append([la, en, edp])
                f.close()

    homo_result_collect = {'fused':[], 'lyl':[]}
    for hw in HW_homo_list:
        for wl in WL_list:
            for sw in schedule_way:
                ky_id = f"{sw}_{hw}_{wl}"
                path = glob.glob(f'{result_path}/*{ky_id}.log')
                assert len(path) == 1, f"Two results found for one design point {ky_id}, something is wrong."
                print(f'Reading in result -- {path}')
                en = None
                la = None
                edp = None
                with open(path[0], 'rb') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        mark = str(line).split()
                        if len(mark) > 20:
                            if mark[3] == "__main__.<module>":
                                en = float(mark[17])
                                la = float(mark[13])
                                edp = float(mark[21][:-3])
                                break
                    if la is None:
                        print(f"{ky_id} not done...")
                    homo_result_collect[sw].append([la, en, edp])
                f.close()

    hetero_result_collect = {'fused':[], 'lyl':[]}
    for hw in HW_hetero_list:
        for wl in WL_list:
            for sw in schedule_way:
                ky_id = f"{sw}_{hw}_{wl}"
                path = glob.glob(f'{result_path}/*{ky_id}.log')
                assert len(path) == 1, f"Two results found for one design point {ky_id}, something is wrong."
                print(f'Reading in result -- {path}')
                en = None
                la = None
                edp = None
                with open(path[0], 'rb') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        mark = str(line).split()
                        if len(mark) > 20:
                            if mark[3] == "__main__.<module>":
                                en = float(mark[17])
                                la = float(mark[13])
                                edp = float(mark[21][:-3])
                                break
                    if la is None:
                        print(f"{ky_id} not done...")
                    hetero_result_collect[sw].append([la, en, edp])
                f.close()


    import matplotlib.pyplot as plt
    import numpy as np

    sc_color = 'red'
    homo_color = 'blue'
    hetero_color = 'green'

    # fig = plt.figure()
    # ax = plt.axes()
    title = ['Latency', 'Energy', 'EDP']
    unit = ['cc', 'pJ', 'cc*pJ']
    plt.figure(figsize=(16,6))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        sc_lyl = np.array([x[i] for x in sc_result_collect['lyl']])
        plt.plot(sc_lyl, color=sc_color, marker='o', label='single-core lbl')
        sc_fused = np.array([x[i] for x in sc_result_collect['fused']])
        plt.plot(sc_fused, linestyle='dotted', color=sc_color, marker='o', label='single-core fused')

        homo_lyl = np.array([x[i] for x in homo_result_collect['lyl']])
        plt.plot(homo_lyl, color=homo_color, marker='o', label='homo lbl')
        homo_fused = np.array([x[i] for x in homo_result_collect['fused']])
        plt.plot(homo_fused, linestyle='dotted', color=homo_color, marker='o', label='homo fused')

        hetero_lyl = np.array([x[i] for x in hetero_result_collect['lyl']])
        plt.plot(hetero_lyl, color=hetero_color, marker='o', label='hetero lbl')
        hetero_fused = np.array([x[i] for x in hetero_result_collect['fused']])
        plt.plot(hetero_fused, linestyle='dotted', color=hetero_color, marker='o', label='hetero fused')

        if i == 0:
            plt.legend()
        plt.ylabel(f"{title[i]} ({unit[i]})")
        plt.xticks([0,1,2,3], ['16x16\n1 core', '32x32\n4 cores', '64x64\n16 cores', '128x128\n64 cores'])
        plt.xlabel("Computation Array Size")
        plt.title(f"{title[i]} ({WL} - opt. for {opt_goal})")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"exploration_figures4/{WL}_{opt_goal}.png")

if __name__ == "__main__":
    for opt_goal in ['latency']:
        result_path = f'/esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results4_{opt_goal}/log/'
        for WL in WL_list:
            plot_cs1(opt_goal,WL,result_path)