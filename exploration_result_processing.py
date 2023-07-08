
import glob
import re
import pickle
import pandas as pd
import xlsxwriter

def command_generator():
    """
    A generator that generates the commands to be run, as strings. Not allowed to end with newline characters
    TODO: write this yourself
    """

    HW_list1 = ['HW1_1bigcore_dpDRAM', 'HW1_1bigcore']
    HW_list2 = ['HW2_4homo_bus', 'HW2_4homo_mesh_dpDRAM', 'HW2_4homo_mesh', 'HW3_4hetero_bus', 'HW3_4hetero_mesh_dpDRAM', 'HW3_4hetero_mesh']
    WL_list = ['resnet18', 'fsrcnn', 'mobilenetv2', 'squeezenet', 'inception_v2']

    experiment_LUT = {}

    i = 0
    for WL in WL_list:
        for HW in HW_list1:
            if i%20 == 0:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log;")
                i += 1
            else:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log & \\")
                i += 1
        for HW in HW_list2:
            if i%20 == 0:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW2_4homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW2_4homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log;")
                i += 1
            else:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW2_4homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW2_4homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log & \\")
                i += 1

    HW_list3 = ["HW500_16homo_mesh_dpDRAM", "HW600_16hetero_mesh_dpDRAM"]
    HW_list4 = ["HW700_64homo_mesh_dpDRAM", "HW800_64hetero_mesh_dpDRAM"]

    i = 80
    for WL in WL_list:
        for HW in HW_list3:
            if i%20 == 0:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW500_16homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW500_16homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log;")
                i += 1
            else:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW500_16homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW500_16homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log & \\")
                i += 1
        for HW in HW_list4:
            if i%20 == 0:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW700_64homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW700_64homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log;")
                i += 1
            else:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW700_64homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW700_64homo &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log & \\")
                i += 1

    HW_list5 = ["single_core_16x16_mesh_dpDRAM", "single_core_32x32_mesh_dpDRAM",
                "single_core_64x64_mesh_dpDRAM", "single_core_128x128_mesh_dpDRAM"]
    i = 120
    for WL in WL_list:
        for HW in HW_list5:
            if i%20 == 0:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log;")
                i += 1
            else:
                experiment_LUT[i] = [WL, HW, 'lbl']
                print(f"python main_stream_exploration_lbl.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_lyl_{HW}_{WL}.log & \\")
                i += 1
                experiment_LUT[i] = [WL, HW, 'fused']
                print(f"python main_stream_exploration_fused.py --headname {i} --workload_path stream/inputs/exploration/workload/{WL}.onnx --accelerator stream.inputs.exploration.hardware.{HW} --mapping_path stream.inputs.exploration.mapping.HW1_1bigcore &> /esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results/log/{i}_fused_{HW}_{WL}.log & \\")
                i += 1

    return experiment_LUT

experiment_LUT = command_generator()


result_collect = []
result_path = '/esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results3_latency/log/'

paths = glob.glob(f'{result_path}/*.log')
for idx, path in enumerate(paths):
    print(f'Reading in result -- {path}')
    ky = int(re.split('[/ _]', path)[12])
    en = None
    la = None
    edp = None
    with open(path, 'rb') as f:
        lines = f.readlines()
        for line in reversed(lines):
            mark = str(line).split()
            if len(mark) > 20:
                if mark[3] == "__main__.<module>":
                    en = float(mark[17])
                    la = float(mark[13])
                    edp = float(mark[21][:-3])
                    break
        result_collect.append([ky]+experiment_LUT[ky]+[la, en, edp])
    f.close()

result_collect = sorted(result_collect, key=lambda x: x[0])

df = pd.DataFrame(result_collect)
writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='result', index=False)
writer.save()

# for result in result_collect:
#     if result[1] == None:
#         print(result)
