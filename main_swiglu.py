import argparse
import logging as _logging
import os
import re

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping_pipelined
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"


def run_main_aie_codegen_swiglu(  # noqa: PLR0913
    seq_len,
    embedding_dim,
    hidden_dim,
    m,
    k,
    n,
    in_dtype,
    out_dtype,
    trace_size,
    rows,
    cols,
    npu,
    line_size,
    runtime_args=None,
):  # noqa: N803, PLR0913
    ############################################INPUTS############################################
    # CREATE THE SWIGLU ONNX MODEL AND MAPPING
    workload_path = make_swiglu_workload(seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype)
    accelerator = os.path.join(os.path.dirname(__file__), "stream/inputs/aie/hardware/whole_array.yaml")
    mapping_path = make_swiglu_mapping_pipelined(seq_len, embedding_dim, hidden_dim, m, k, n, line_size)
    mode = "fused"
    layer_stacks = [(0, 1, 2, 3, 4)]
    if runtime_args is None:
        runtime_args = ["input", "weight_1", "weight_2", "weight_3", "output"]
    ##############################################################################################

    ################################PARSING###############################
    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    mapping_name = f"{rows}_row_{cols}_col"
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
    logger.info(
        f"Running AIE code generation for Swiglu with "
        f"seq_len={seq_len}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}"
    )
    ######################################################################

    ################################PLOTS################################
    # memory_fig_path = f"outputs/{experiment_id}/memory.png"
    # json_path = f"outputs/{experiment_id}/scme.json"
    #####################################################################

    module = optimize_allocation_co(
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
        runtime_args=runtime_args,
    )

    # #####################CostModelEvaluationLUT LOAD#############################
    # cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
    # cost_lut = CostModelEvaluationLUT(cost_lut_path)
    # #############################################################################

    # # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    # convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

    # # Plotting memory usage of best SCME
    # plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIE code generation for Gemm")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length (seq_len dimension of the input)")
    parser.add_argument(
        "--embedding_dim", type=int, required=True, help="Embedding dimension (embedding_dim dimension of the input)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, required=True, help="Hidden dimension (hidden_dim dimension of the output)"
    )
    parser.add_argument("--line_size", type=int, required=True, help="N parameter for the model")
    parser.add_argument("--m", type=int, default=32, help="m parameter for the model (default: 32)")
    parser.add_argument("--k", type=int, default=32, help="k parameter for the model (default: 32)")
    parser.add_argument("--n", type=int, default=32, help="n parameter for the model (default: 32)")
    parser.add_argument("--in_dtype", type=str, default="bf16", help="Input data type (default: bf16)")
    parser.add_argument("--out_dtype", type=str, default="bf16", help="Output data type (default: bf16)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Size of the trace buffer (default: 1048576)")
    parser.add_argument("--rows", type=int, default=4, help="Number of AIE rows to use (has to be 4)")
    parser.add_argument("--cols", type=int, default=1, help="Number of AIE columns to use (default: 1)")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target (default: npu2)")
    args = parser.parse_args()
    assert args.cols == 1, (
        "This script only supports 1 AIE column. Use main_gemm_whole_array.py for more than 1 column."
    )

    module = run_main_aie_codegen_swiglu(
        args.seq_len,
        args.embedding_dim,
        args.hidden_dim,
        args.m,
        args.k,
        args.n,
        args.in_dtype,
        args.out_dtype,
        args.trace_size,
        args.rows,
        args.cols,
        args.npu,
        args.line_size,
    )

    print(str(module))
