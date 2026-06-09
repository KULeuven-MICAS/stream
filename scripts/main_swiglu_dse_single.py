import argparse
import logging as _logging
import os
import re

from stream.api import configure_logging, optimize_allocation_co
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.opt.solver import ConstraintSelection

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"


def run_main_aie_codegen_swiglu(  # noqa: PLR0913
    seq_len,
    embedding_dim,
    hidden_dim,
    in_dtype,
    out_dtype,
    trace_size,
    rows,
    cols,
    npu,
    seq_len_tile_size=1,
    embedding_tile_size=512,
    hidden_tile_size=64,
    last_gemm_down: bool = False,
    constraint_selection: ConstraintSelection | None = None,
):  # noqa: N803, PLR0913
    ############################################INPUTS############################################
    accelerator = os.path.join(os.path.dirname(__file__), "../stream/inputs/aie/hardware/whole_array_strix.yaml")
    # CREATE THE SWIGLU ONNX MODEL AND MAPPING
    workload_path = make_swiglu_workload(
        seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype, last_gemm_down=last_gemm_down
    )
    mapping_path = (
        "outputs/dse-20260429-whole_array_strix-swiglu_256_2048_8192-4_row_8_col"
        "/tilesizes_32_128_32/42/swiglu_256_2048_2048_mapping.yaml"
    )
    ##############################################################################################

    ################################PARSING###############################
    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    mapping_name = f"{rows}_row_{cols}_col"
    experiment_id = f"dse-single-{hw_name}-{wl_name}-{mapping_name}"
    ######################################################################

    ################################LOGGING###############################
    log_path = os.path.join(os.getcwd(), f"outputs/{experiment_id}/stream.log")
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

    ctx = optimize_allocation_co(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
        enable_codegen=True,
        trace_size=trace_size,
        nb_cols_to_use=cols,
        npu=npu,
        constraint_selection=constraint_selection,
    )

    module = ctx.get("module")

    return module


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Run AIE code generation for Gemm")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length (seq_len dimension of the input)")
    parser.add_argument(
        "--embedding_dim", type=int, required=True, help="Embedding dimension (embedding_dim dimension of the input)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, required=True, help="Hidden dimension (hidden_dim dimension of the output)"
    )
    parser.add_argument("--in_dtype", type=str, default="bf16", help="Input data type (default: bf16)")
    parser.add_argument("--out_dtype", type=str, default="bf16", help="Output data type (default: bf16)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Size of the trace buffer (default: 1048576)")
    parser.add_argument("--rows", type=int, default=4, help="Number of AIE rows to use (has to be 4)")
    parser.add_argument("--cols", type=int, default=8, help="Number of AIE columns to use (default: 8)")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target (default: npu2)")
    parser.add_argument(
        "--seq_len_tile_size", type=int, default=32, help="Tile size for seq_len dimension (default: 32)"
    )
    parser.add_argument(
        "--embedding_tile_size", type=int, default=128, help="Tile size for embedding dimension (default: 128)"
    )
    parser.add_argument("--hidden_tile_size", type=int, default=64, help="Tile size for hidden dimension (default: 64)")
    parser.add_argument(
        "--no_last_gemm_down",
        dest="last_gemm_down",
        action="store_false",
        default=True,
        help="If set, the last gemm down projection is skipped",
    )
    parser.add_argument(
        "--disable-constraints",
        nargs="*",
        choices=["memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels"],
        default=[],
        metavar="CONSTRAINT",
        help="Disable hardware resource constraint groups. Example: --disable-constraints memory_capacity dma_channels",
    )
    args = parser.parse_args()
    _disabled = set(args.disable_constraints or [])
    _constraint_selection = (
        ConstraintSelection(
            memory_capacity="memory_capacity" not in _disabled,
            object_fifo_depth="object_fifo_depth" not in _disabled,
            buffer_descriptors="buffer_descriptors" not in _disabled,
            dma_channels="dma_channels" not in _disabled,
        )
        if _disabled
        else None
    )
    module = run_main_aie_codegen_swiglu(
        args.seq_len,
        args.embedding_dim,
        args.hidden_dim,
        args.in_dtype,
        args.out_dtype,
        args.trace_size,
        args.rows,
        args.cols,
        args.npu,
        args.seq_len_tile_size,
        args.embedding_tile_size,
        args.hidden_tile_size,
        last_gemm_down=args.last_gemm_down,
        constraint_selection=_constraint_selection,
    )
    save_path = f"outputs/swiglu_module_{args.seq_len}_{args.embedding_dim}_{args.hidden_dim}.mlir"
    with open(save_path, "w") as f:
        f.write(str(module))
    print(f"Saved generated module to {save_path}")
