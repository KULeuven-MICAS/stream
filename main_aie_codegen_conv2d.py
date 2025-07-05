import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json
from stream.inputs.aie.workload.make_conv2d_onnx import make_conv2d
import argparse

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

def run_main_aie_codegen(H):
    ############################################INPUTS############################################
    # CREATE THE CONV ONNX MODEL
    workload_path = make_conv2d(H)
    accelerator = "stream/inputs/aie/hardware/single_aie_tile.yaml"
    mapping_path = "stream/inputs/aie/mapping/single_aie_tile.yaml"
    # mode = "lbl"
    # layer_stacks = [(0,),]
    mode = "fused"
    layer_stacks = [(0,)]
    ##############################################################################################

    ################################PARSING###############################
    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}-{mode}-constraint-optimization"
    ######################################################################

    ##############PLOTTING###############
    draw_dependencies = True
    plot_data_transfer = True
    section_start_percent = (0,)
    percent_shown = (100,)
    #####################################


    ################################PATHS################################
    timeline_fig_path_plotly = f"outputs/{experiment_id}/schedule.html"
    memory_fig_path = f"outputs/{experiment_id}/memory.png"
    json_path = f"outputs/{experiment_id}/scme.json"
    #####################################################################

    scme = optimize_allocation_co(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
        enable_codegen=True,
    )

    #####################CostModelEvaluationLUT LOAD#############################
    cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    #############################################################################
    
    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

    # Plotting memory usage of best SCME
    plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIE code generation")
    parser.add_argument("--height", type=int, required=True, help="Height parameter for the model")
    args = parser.parse_args()

    run_main_aie_codegen(args.height)