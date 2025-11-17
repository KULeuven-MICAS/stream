import argparse
import logging as _logging
import os
import re

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping_pipelined
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"


def run_main_aie_codegen_swiglu(M, K, N, m, k, n, line_size, in_dtype, out_dtype, trace_size, nb_rows, nb_cols, npu):  # noqa: N803, PLR0913
    ############################################INPUTS############################################
    # CREATE THE CONV ONNX MODEL
    input_shape = (M, K)
    out_channels = N
    workload_path = make_swiglu_workload(input_shape, out_channels)
    accelerator = "stream/inputs/aie/hardware/single_core.yaml"
    mapping_path = make_swiglu_mapping_pipelined(input_shape, out_channels, m, k, n, line_size)
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
    mapping_name = f"{nb_rows}_row_{nb_cols}_col"
    experiment_id = f"{hw_name}-{wl_name}-{mapping_name}"
    ######################################################################

    ################################LOGGING###############################
    log_path = f"outputs/{experiment_id}/stream.log"
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Get root logger and remove any existing handlers
    logger = _logging.getLogger()
    logger.setLevel(_logging_level)  # or use _logging_level if you define one
    # Remove all existing handlers (e.g., ones added by Snakemake or libraries)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Create a file handler explicitly
    file_handler = _logging.FileHandler(log_path)
    file_handler.setFormatter(_logging.Formatter(_logging_format))
    logger.addHandler(file_handler)
    logger.info(f"Running AIE code generation for Gemm with M={M}, N={N}, K={K}")
    ######################################################################

    ################################PLOTS################################
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
        npu=npu,
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
    parser.add_argument("--K", type=int, required=True, help="K parameter for the model")
    parser.add_argument("--N", type=int, required=True, help="N parameter for the model")
    parser.add_argument("--m", type=int, default=32, help="m parameter for the model (default: 32)")
    parser.add_argument("--k", type=int, default=32, help="k parameter for the model (default: 32)")
    parser.add_argument("--n", type=int, default=32, help="n parameter for the model (default: 32)")
    parser.add_argument("--in_dtype", type=str, default="i16", help="Input data type (default: i16)")
    parser.add_argument("--out_dtype", type=str, default="i32", help="Output data type (default: i32)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Size of the trace buffer (default: 1048576)")
    parser.add_argument("--rows", type=int, default=1, help="Number of AIE rows to use (has to be 1)")
    parser.add_argument("--cols", type=int, default=1, help="Number of AIE columns to use (default: 1)")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target (default: npu2)")
    args = parser.parse_args()
    assert args.rows == 1, "This script only supports 1 AIE row. Use main_gemm_single_col.py for more than 1 row."
    assert args.cols == 1, (
        "This script only supports 1 AIE column. Use main_gemm_whole_array.py for more than 1 column."
    )

    run_main_aie_codegen_swiglu(
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.in_dtype,
        args.out_dtype,
        args.trace_size,
        args.rows,
        args.cols,
        args.npu,
    )
