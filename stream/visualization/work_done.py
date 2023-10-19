from typing import List
import numpy as np
import plotly.express as px
import pandas as pd

from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.utils import load_scme

def plot_work_done(scmes: List[StreamCostModelEvaluation], nb_x_ticks, max_latency, cores_for_ideal=()):
    df_all = pd.DataFrame(columns=["timestep", "ops_done_percentage", "ops_done", "accelerator", "type"])
    for scme in scmes:
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
        if cores_for_ideal:
            timesteps_ideal = []
            ops_done_ideal = []
            percentages_ideal = []
            acc_names_ideal = []
            types_ideal = []
            ideal_ops_per_cycle = sum([scme.accelerator.get_core(core_id).operational_array.total_unit_count for core_id in cores_for_ideal])
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
    fig.write_html("outputs/exploration_k_split/work_done_ops.html")

if __name__ == "__main__":
    scme122 = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/122-single_core_32x32_mesh_dpDRAM-resnet18-hintloop_-lbl-scme.pickle')
    scme7 = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/7-HW2_4homo_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme.pickle')
    scme13 = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/13-HW3_4hetero_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme.pickle')
    # scme13_half = load_scme('/users/micas/asymons/stream_TC_2023/test_oy_half/inter_result/13-HW3_4hetero_mesh_dpDRAM-resnet18-hintloop_oy_all-fused-scme.pickle')
    scmes = [scme122, scme7, scme13]
    # scmes = [scme13, scme13_half]

    # compare mobilenetv2 results: 1 32x32 core lbl vs 4 16x16 core heterogeneous lbl
    # scme_mobilenetv2_single = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/138-single_core_32x32_mesh_dpDRAM-mobilenetv2-hintloop_-lbl-scme.pickle')
    # scme_mobilenetv2_quad_hetero_lbl = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/44-HW3_4hetero_mesh_dpDRAM-mobilenetv2-hintloop_-lbl-scme.pickle')
    # scme_mobilenetv2_quad_hetero_fused = load_scme('/users/micas/asymons/stream_TC_2023/exploration_k_split_memory/inter_result/45-HW3_4hetero_mesh_dpDRAM-mobilenetv2-hintloop_oy_all-fused-scme.pickle')
    # scmes = [scme_mobilenetv2_single, scme_mobilenetv2_quad_hetero_lbl, scme_mobilenetv2_quad_hetero_fused]
    max_latency = max([scme.latency for scme in scmes])
    plot_work_done(scmes, nb_x_ticks=100, max_latency=max_latency)  #, cores_for_ideal=[0,1,2,3])