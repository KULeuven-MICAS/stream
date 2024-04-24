from typing import List
import numpy as np
import plotly.express as px
import pandas as pd

from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.utils import load_scme

def plot_work_done(scmes: List[StreamCostModelEvaluation], nb_x_ticks, max_latency, cores_for_ideal=[], fig_path="outputs/exploration_co/work_done_ops.html"):
    df_all = pd.DataFrame(columns=["timestep", "ops_done_percentage", "ops_done", "accelerator", "type"])
    if not cores_for_ideal:
        cores_for_ideal = [tuple() for _ in range(len(scmes))]
    assert len(scmes) == len(cores_for_ideal)
    for scme, for_ideal in zip(scmes, cores_for_ideal):
        ## REAL
        acc_name = scme.accelerator.name
        workload = scme.workload
        total_ops = sum([n.total_MAC_count for n in workload.nodes()])
        ends, cns = zip(*sorted((cn.end, cn) for cn in workload.nodes()))
        all_timesteps = [(i/(nb_x_ticks-1)) * max_latency for i in range(nb_x_ticks)]
        timesteps = []
        ops_done = []
        percentages = []
        acc_names = []
        types = []
        for timestep in all_timesteps:
            timesteps.append(timestep)
            ends_idx = np.searchsorted(ends, timestep, side='right')
            cns_done = cns[:ends_idx]
            ops = sum([cn.total_MAC_count for cn in cns_done])
            percentage_done = ops/total_ops
            ops_done.append(ops)
            percentages.append(percentage_done)
            acc_names.append(acc_name)
            types.append("real")
            if ops == total_ops:
                break
        df = pd.DataFrame({"timestep": timesteps, "ops_done_percentage": percentages, "ops_done": ops_done, "accelerator": acc_names, "type": types})
        df_all = pd.concat([df_all, df], ignore_index=True)
        ## IDEAL
        if for_ideal:
            timesteps_ideal = []
            ops_done_ideal = []
            percentages_ideal = []
            acc_names_ideal = []
            types_ideal = []
            ideal_ops_per_cycle = sum([scme.accelerator.get_core(core_id).operational_array.total_unit_count for core_id in for_ideal])
            for timestep in all_timesteps:
                ops = ideal_ops_per_cycle * timestep
                if ops > total_ops:
                    ops = total_ops
                timesteps_ideal.append(timestep)
                ops_done_ideal.append(ops)
                percentages_ideal.append(ops/total_ops)
                acc_names_ideal.append(acc_name)
                types_ideal.append("ideal")
                if ops == total_ops:
                    break
            df_ideal = pd.DataFrame({"timestep": timesteps_ideal, "ops_done_percentage": percentages_ideal, "ops_done": ops_done_ideal, "accelerator": acc_names_ideal, "type": types_ideal})
            df_all = pd.concat([df_all, df_ideal], ignore_index=True)



    fig = px.line(df_all, x='timestep', y='ops_done', color='accelerator', symbol='type')
    fig.write_html(fig_path)
    print(f"Work done visualization saved to: {fig_path}")

if __name__ == "__main__":
    scme122 = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/122-single_core_32x32_mesh_dpDRAM-resnet18-hintloop_-lbl-scme.pickle')
    scme7 = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/7-HW2_4homo_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme.pickle')
    scme13 = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/13-HW3_4hetero_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme.pickle')
    scmes = [scme122, scme7, scme13]
    # scmes = [scme13, scme13_half]
    scme81 = load_scme("/users/micas/asymons/stream_TC_2023/exploration_co_multiple_stacks_new_cost_model/inter_result/81-HW500_16homo_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme-with-communication.pickle")
    scme124 = load_scme("/users/micas/asymons/stream_TC_2023/exploration_co_multiple_stacks_new_cost_model/inter_result/124-single_core_64x64_mesh_dpDRAM-resnet18-hintloop_-lbl-scme-with-communication.pickle")
    scme125 = load_scme("/users/micas/asymons/stream_TC_2023/exploration_co_multiple_stacks_new_cost_model/inter_result/125-single_core_64x64_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme-with-communication.pickle")
    scmes = [scme81, scme125,]

    max_latency = max([scme.latency for scme in scmes])
    cores_for_ideal = [tuple(range(16)), (0,)]
    fig_path = "outputs/exploration_co/work_done_ops_81_125.html"
    plot_work_done(scmes, nb_x_ticks=100, max_latency=max_latency, cores_for_ideal=cores_for_ideal, fig_path=fig_path)