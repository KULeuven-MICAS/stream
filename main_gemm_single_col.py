import argparse
import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping_single_col
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def run_main_aie_codegen_gemm(M, K, N, m, k, n, in_dtype, out_dtype, trace_size):  # noqa: N803, PLR0913
    ############################################INPUTS############################################
    # CREATE THE CONV ONNX MODEL
    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    accelerator = "stream/inputs/aie/hardware/single_col.yaml"
    mapping_path = make_gemm_mapping_single_col(M, K, N, m, k, n, has_mem_tile=True, nb_compute_cores=2)
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
    # section_start_percent = (0,)
    # percent_shown = (100,)
    #####################################

    ################################PATHS################################
    # memory_fig_path = f"outputs/{experiment_id}/memory.png"
    # json_path = f"outputs/{experiment_id}/scme.json"
    #####################################################################

    _ = optimize_allocation_co(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
        enable_codegen=True,
        trace_size=trace_size,
    )

    # #####################CostModelEvaluationLUT LOAD#############################
    # cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
    # cost_lut = CostModelEvaluationLUT(cost_lut_path)
    # #############################################################################

    # # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    # convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

    # # Plotting memory usage of best SCME
    # plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIE code generation for Gemm")
    parser.add_argument("--M", type=int, required=True, help="M parameter for the model")
    parser.add_argument("--N", type=int, required=True, help="N parameter for the model")
    parser.add_argument("--K", type=int, required=True, help="K parameter for the model")
    parser.add_argument("--m", type=int, default=32, help="m parameter for the model (default: 32)")
    parser.add_argument("--k", type=int, default=32, help="k parameter for the model (default: 32)")
    parser.add_argument("--n", type=int, default=32, help="n parameter for the model (default: 32)")
    parser.add_argument("--in_dtype", type=str, default="i16", help="Input data type (default: i16)")
    parser.add_argument("--out_dtype", type=str, default="i32", help="Output data type (default: i32)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Size of the trace buffer (default: 1048576)")
    args = parser.parse_args()

    run_main_aie_codegen_gemm(
        args.M, args.K, args.N, args.m, args.k, args.n, args.in_dtype, args.out_dtype, args.trace_size
    )
