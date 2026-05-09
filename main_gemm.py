import argparse
import logging as _logging
import os
import re

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.opt.solver import ConstraintSelection

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def run_main_aie_codegen_gemm(
    M,
    K,
    N,
    m,
    k,
    n,
    in_dtype,
    out_dtype,
    trace_size,
    nb_rows,
    nb_cols,
    npu,
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
):  # noqa: N803, PLR0913
    ############################################INPUTS############################################
    # CREATE THE CONV ONNX MODEL
    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    accelerator = os.path.join(os.path.dirname(__file__), "stream/inputs/aie/hardware/whole_array_strix.yaml")
    mapping_path = make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use=nb_rows, nb_cols_to_use=nb_cols)
    ##############################################################################################

    ################################PARSING###############################
    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    mapping_name = f"{nb_rows}_row_{nb_cols}_col"
    experiment_id = f"{hw_name}-{wl_name}-{mapping_name}"
    ######################################################################

    ##############PLOTTING###############
    # section_start_percent = (0,)
    # percent_shown = (100,)
    #####################################

    ################################PATHS################################
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
        nb_cols_to_use=nb_cols,
        npu=npu,
        backend=backend,
        constraint_selection=constraint_selection,
    )

    module = ctx.get("module")

    # Save the mlir module to output.mlir
    mlir_path = f"outputs/{experiment_id}/output.mlir"
    with open(mlir_path, "w") as f:
        f.write(str(module))

    return module


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
    parser.add_argument("--rows", type=int, default=2, help="Number of AIE rows to use (default: 2)")
    parser.add_argument("--cols", type=int, default=2, help="Number of AIE columns to use (default: 2)")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target (default: npu2)")
    parser.add_argument(
        "--backend",
        type=str,
        default="ortools_gscip",
        choices=["gurobi", "ortools_gscip", "ortools_highs", "ortools_gurobi"],
        help="Solver backend (default: ortools_gscip)",
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

    module = run_main_aie_codegen_gemm(
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
        args.backend,
        constraint_selection=_constraint_selection,
    )
    save_path = f"outputs/swiglu_module_{args.M}_{args.N}_{args.K}.mlir"
    with open(save_path, "w") as f:
        f.write(str(module))
    print(f"Saved generated module to {save_path}")
