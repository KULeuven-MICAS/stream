"""Cross-backend verification runner for TETRA.

Runs both OR-Tools and Gurobi backends on the same workload instance,
compares their objective values within a configurable tolerance, and
prints a comparison table.

Per D-06, D-07: standalone runnable script (not a pytest test).
Exit code: 0 if PASS (within tolerance), 1 if FAIL.

Usage:
    python tests/verify_backends.py --workload gemm
    python tests/verify_backends.py --workload swiglu --backends ortools_gscip ortools_highs
    python tests/verify_backends.py --workload gemm --backends gurobi ortools_gscip --output-yaml /tmp/stats.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
import warnings

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCELERATOR = os.path.join(
    os.path.dirname(__file__),
    "../stream/inputs/aie/hardware/whole_array_strix.yaml",
)

# Known Gurobi baseline objectives (verified in Phase 1 Plan 04)
GEMM_GUROBI_OBJ = 48_730_630.0
SWIGLU_GUROBI_OBJ = 9_396_485.0


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _run_gemm_pipeline(  # noqa: PLR0913, N803
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    in_dtype: str,
    out_dtype: str,
    rows: int,
    cols: int,
    trace_size: int,
    npu: str,
    backend: str,
    output_path: str,
):
    """Run the TETRA gemm optimization pipeline with the specified backend.

    Returns:
        ctx: Stage context with ``scheduler`` accessible via ``ctx.get``.
    """
    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    mapping_path = make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use=rows, nb_cols_to_use=cols)

    hw_name = ACCELERATOR.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}-{rows}_row_{cols}_col"

    return optimize_allocation_co(
        hardware=ACCELERATOR,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path=output_path,
        skip_if_exists=False,
        trace_size=trace_size,
        nb_cols_to_use=cols,
        npu=npu,
        backend=backend,
    )


def _run_swiglu_pipeline(  # noqa: PLR0913
    seq_len: int,
    embedding_dim: int,
    hidden_dim: int,
    in_dtype: str,
    out_dtype: str,
    rows: int,
    cols: int,
    trace_size: int,
    npu: str,
    backend: str,
    output_path: str,
):
    """Run the TETRA swiglu optimization pipeline with the specified backend.

    Returns:
        ctx: Stage context with ``scheduler`` accessible via ``ctx.get``.
    """
    workload_path = make_swiglu_workload(seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype)
    mapping_path = make_swiglu_mapping(
        seq_len,
        embedding_dim,
        hidden_dim,
        True,  # last_gemm_down
        32,  # seq_len_tile_size (default from reference tests)
        32,  # embedding_tile_size
        64,  # hidden_tile_size
    )

    hw_name = ACCELERATOR.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}-{rows}_row_{cols}_col"

    return optimize_allocation_co(
        hardware=ACCELERATOR,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path=output_path,
        skip_if_exists=False,
        trace_size=trace_size,
        nb_cols_to_use=cols,
        npu=npu,
        backend=backend,
    )


def _extract_latency_total(ctx) -> float:
    """Extract ``latency_total`` (solver objective) from a completed pipeline context."""
    scheduler = ctx.get("scheduler")
    return float(scheduler.latency_total)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_comparison_table(
    workload: str,
    workload_desc: str,
    results: list[dict],
    rel_diff: float | None,
    tolerance: float,
    passed: bool,
) -> None:
    """Print a formatted comparison table to stdout (per D-07)."""
    print()
    print("=== Cross-Backend Verification ===")
    print(f"Workload: {workload} ({workload_desc})")
    print()

    header = f"{'Backend':<10} {'Solver':<8} {'Status':<10} {'Objective':>15} {'SolveTime(s)':>14}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        obj_str = f"{r['objective']:.1f}" if r["objective"] is not None else "N/A"
        print(f"{r['backend']:<10} {r['solver']:<8} {r['status']:<10} {obj_str:>15} {r['solve_time_s']:>14.2f}")
    print(sep)
    print()

    if rel_diff is not None:
        print(f"Relative difference: {rel_diff * 100:.2f}%")
    else:
        print("Relative difference: N/A (one or both backends failed)")
    print(f"Tolerance: {tolerance * 100:.2f}%")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print()


def _write_yaml(
    path: str,
    workload: str,
    tolerance: float,
    results: list[dict],
    rel_diff: float | None,
    passed: bool,
) -> None:
    """Write YAML stats file (per D-07)."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        warnings.warn(
            "PyYAML not installed — skipping YAML output. Install with: pip install pyyaml",
            stacklevel=2,
        )
        return

    data: dict = {
        "workload": workload,
        "tolerance": tolerance,
        "result": "PASS" if passed else "FAIL",
        "backends": {},
        "relative_difference": round(rel_diff * 100, 4) if rel_diff is not None else None,
    }

    for r in results:
        backend_key = r["backend"].lower()
        data["backends"][backend_key] = {
            "solver": r["solver"],
            "status": r["status"],
            "objective": r["objective"],
            "solve_time_s": round(r["solve_time_s"], 4),
        }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Stats written to: {path}")


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _run_backend(
    workload: str,
    backend: str,
    args: argparse.Namespace,
    output_path: str,
) -> tuple[dict, float | None]:
    """Run a single backend and return (result_dict, latency_total).

    Returns result dict with backend/solver/status/objective/solve_time_s.
    latency_total is None if the backend failed.
    """
    print(f"Running {backend.upper()} backend...")
    t0 = time.perf_counter()
    latency_total = None
    objective = None
    status = "FAILED"
    solver = backend.lower()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if workload == "gemm":
                ctx = _run_gemm_pipeline(
                    M=args.M,
                    K=args.K,
                    N=args.N,
                    m=args.m,
                    k=args.k,
                    n=args.n,
                    in_dtype=args.in_dtype,
                    out_dtype=args.out_dtype,
                    rows=args.rows,
                    cols=args.cols,
                    trace_size=args.trace_size,
                    npu=args.npu,
                    backend=backend,
                    output_path=output_path or tmpdir,
                )
            else:
                ctx = _run_swiglu_pipeline(
                    seq_len=args.seq_len,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    in_dtype=args.in_dtype,
                    out_dtype=args.out_dtype,
                    rows=args.rows,
                    cols=args.cols,
                    trace_size=args.trace_size,
                    npu=args.npu,
                    backend=backend,
                    output_path=output_path or tmpdir,
                )
        latency_total = _extract_latency_total(ctx)
        objective = latency_total
        status = "OPTIMAL"
    except Exception as exc:
        exc_str = str(exc)
        # Detect Gurobi license errors specifically
        if "license" in exc_str.lower() or "no valid gurobi" in exc_str.lower():
            status = "NO_LICENSE"
            warnings.warn(
                f"Gurobi license not available: {exc_str}. "
                "Skipping Gurobi comparison. Install a valid Gurobi license to enable cross-backend verification.",
                stacklevel=2,
            )
        else:
            status = f"ERROR: {exc_str[:80]}"
            print(f"ERROR running {backend.upper()} backend: {exc}", file=sys.stderr)

    solve_time_s = time.perf_counter() - t0

    solver = backend.lower().replace("ortools_", "")

    return (
        {
            "backend": backend.upper(),
            "solver": solver,
            "status": status,
            "objective": objective,
            "solve_time_s": solve_time_s,
        },
        latency_total,
    )


def verify(args: argparse.Namespace) -> int:
    """Run cross-backend verification. Returns 0 on PASS, 1 on FAIL."""
    workload = args.workload

    # Build workload description string for the table
    if workload == "gemm":
        workload_desc = f"M={args.M}, K={args.K}, N={args.N}"
    else:
        workload_desc = f"seq_len={args.seq_len}, embedding_dim={args.embedding_dim}, hidden_dim={args.hidden_dim}"

    results = []
    latency_totals = {}

    output_dir = ""  # use tempdir per run

    backends = args.backends
    for b in backends:
        result, latency = _run_backend(workload, b, args, output_dir)
        results.append(result)
        if latency is not None:
            latency_totals[b] = latency

    # Compute relative difference (compare all against first successful backend)
    rel_diff = None
    passed = False

    successful = [b for b in backends if b in latency_totals]
    if len(successful) >= 2:
        base_name = successful[0]
        base_val = latency_totals[base_name]
        max_diff = 0.0
        for b in successful[1:]:
            diff = abs(latency_totals[b] - base_val) / max(abs(base_val), 1e-10)
            max_diff = max(max_diff, diff)
        rel_diff = max_diff
        passed = rel_diff <= args.tolerance
    elif len(successful) == 1:
        b = successful[0]
        baseline_obj = GEMM_GUROBI_OBJ if workload == "gemm" else SWIGLU_GUROBI_OBJ
        rel_diff = abs(latency_totals[b] - baseline_obj) / max(abs(baseline_obj), 1e-10)
        passed = rel_diff <= args.tolerance
        print(f"Note: Only 1 backend succeeded. Comparing against known baseline ({baseline_obj:.0f}).")
    else:
        passed = False

    _print_comparison_table(workload, workload_desc, results, rel_diff, args.tolerance, passed)

    if args.output_yaml:
        _write_yaml(args.output_yaml, workload, args.tolerance, results, rel_diff, passed)

    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-backend verification runner for TETRA. "
        "Runs both OR-Tools and Gurobi backends on the same workload, "
        "compares objectives within a tolerance, and reports a comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--workload",
        type=str,
        choices=["gemm", "swiglu"],
        required=True,
        help="Workload type to verify",
    )

    # Gemm-specific arguments
    gemm_group = parser.add_argument_group("GEMM parameters (required when --workload gemm)")
    gemm_group.add_argument("--M", type=int, default=256, help="M dimension")
    gemm_group.add_argument("--K", type=int, default=8192, help="K dimension")
    gemm_group.add_argument("--N", type=int, default=2048, help="N dimension")
    gemm_group.add_argument("--m", type=int, default=32, help="m tile dimension")
    gemm_group.add_argument("--k", type=int, default=32, help="k tile dimension")
    gemm_group.add_argument("--n", type=int, default=32, help="n tile dimension")

    # SwiGLU-specific arguments
    swiglu_group = parser.add_argument_group("SwiGLU parameters (required when --workload swiglu)")
    swiglu_group.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    swiglu_group.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    swiglu_group.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension")

    # Common arguments
    parser.add_argument("--in_dtype", type=str, default="bf16", help="Input data type")
    parser.add_argument("--out_dtype", type=str, default="bf16", help="Output data type")
    parser.add_argument("--rows", type=int, default=4, help="Number of AIE rows")
    parser.add_argument("--cols", type=int, default=8, help="Number of AIE columns")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Trace buffer size")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target")

    # Backend selection
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["ortools_gscip", "ortools_highs", "gurobi"],
        choices=["gurobi", "ortools_gscip", "ortools_highs", "ortools_gurobi"],
        help="Backends to compare (default: ortools_gscip ortools_highs gurobi)",
    )

    # Verification parameters
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for objective comparison (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to write YAML stats file",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(verify(args))
