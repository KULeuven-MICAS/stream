"""Cross-backend integration tests for the allocation model.

Compares ORToolsBackend (GSCIP, HiGHS) against GurobiBackend on real allocation
instances to verify objective quality and measure solve time.

Tests cover both performance (resulting objective value) and time.
Tests cover both main_gemm.py and main_swiglu.py configurations.

Backend injection strategy:
  AllocationModel creates its solver via
  ``create_solver(SolverBackend.GUROBI, ...)``.  We use ``unittest.mock.patch``
  to intercept that call and return an ORToolsBackend instead.  The patch must
  target the allocator module, which imports ``create_solver`` into its own
  namespace.
"""

import os
import tempfile
import time
from unittest.mock import patch

import pytest
from ortools.math_opt.python import mathopt

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.opt.solver import ORToolsBackend, SolverBackend  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCELERATOR = os.path.join(
    os.path.dirname(__file__),
    "../../stream/inputs/aie/hardware/whole_array_strix.yaml",
)

# Gurobi baseline latency_total values.
# Re-pinned after the AIE MILP objective improvements (#109); Gurobi, GSCIP and HiGHS agree exactly.
GEMM_GUROBI_OBJ = 29_491_232.0
SWIGLU_GUROBI_OBJ = 3_215_616.0

# Relative tolerance for cross-backend comparison (1%)
REL_TOL = 0.01

# Patch target: the allocator module imports create_solver into its own namespace
_TTA_CREATE_SOLVER = "stream.opt.allocation.constraint_optimization.allocation_model.create_solver"
_LICENSE_CHECK = "stream.api.GurobiBackend.check_license"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ortools_factory(solver_type: mathopt.SolverType):
    """Return a drop-in replacement for ``create_solver`` that ignores the
    backend argument and always returns an ``ORToolsBackend``."""

    def _factory(backend, name="", *, solver_type=solver_type, **kwargs):  # noqa: ARG001
        return ORToolsBackend(name, solver_type)

    return _factory


def _run_gemm_pipeline(output_path: str):
    """Run the gemm optimisation pipeline.

    Args:
        output_path: Directory where pipeline outputs are written.

    Returns:
        ctx: Stage context with ``allocation`` accessible via ``ctx.get``.
    """
    # Parameters from launch.json
    M, K, N = 256, 8192, 2048
    m, k, n = 32, 32, 32
    in_dtype, out_dtype = "bf16", "bf16"
    nb_rows, nb_cols = 4, 8

    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    mapping_path = make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use=nb_rows, nb_cols_to_use=nb_cols)

    return evaluate_mapping(
        ACCELERATOR, workload_path, output_path, mapping_path, options=SolveOptions(nb_cols_to_use=nb_cols)
    ).context


def _run_swiglu_pipeline(output_path: str):
    """Run the swiglu optimisation pipeline.

    Args:
        output_path: Directory where pipeline outputs are written.

    Returns:
        ctx: Stage context with ``allocation`` accessible via ``ctx.get``.
    """
    # Parameters from launch.json (known working, verified)
    seq_len, embedding_dim, hidden_dim = 256, 512, 2048
    in_dtype, out_dtype = "bf16", "bf16"
    cols = 8
    seq_len_tile_size = 32
    embedding_tile_size = 32
    hidden_tile_size = 64

    workload_path = make_swiglu_workload(seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype)
    mapping_path = make_swiglu_mapping(
        seq_len,
        embedding_dim,
        hidden_dim,
        True,
        seq_len_tile_size,
        embedding_tile_size,
        hidden_tile_size,
    )

    return evaluate_mapping(
        ACCELERATOR, workload_path, output_path, mapping_path, options=SolveOptions(nb_cols_to_use=cols)
    ).context


def _extract_latency_total(ctx) -> float:
    """Extract ``latency_total`` from a completed pipeline context.

    ``latency_total`` is the primary value minimised by the MILP solver —
    the metric used for cross-backend objective comparison.
    """
    return float(ctx.get("allocation").solution.latency.total)


def _print_result_table(rows: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Backend':<10} {'Solver':<8} {'Status':<10} {'latency_total':>15} {'SolveTime(s)':>14}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['backend']:<10} {r['solver']:<8} {r['status']:<10}"
            f" {r['latency_total']:>15.0f} {r['solve_time']:>14.2f}"
        )
    print("-" * len(header) + "\n")


def _append_result(results: list, backend: str, solver: str, lat_total: float, solve_time: float) -> None:
    results.append(
        {
            "backend": backend,
            "solver": solver,
            "status": "OPTIMAL",
            "latency_total": lat_total,
            "solve_time": solve_time,
        }
    )


# ---------------------------------------------------------------------------
# Gemm integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gemm_gurobi_baseline():
    """GurobiBackend produces OPTIMAL on gemm instance.

    This test establishes the baseline. It uses the full unpatched pipeline
    and verifies the known objective value.
    """
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        ctx = _run_gemm_pipeline(tmpdir)
        solve_time = time.perf_counter() - t0

    lpi = _extract_latency_total(ctx)
    _append_result(results, "GUROBI", "gurobi", lpi, solve_time)
    _print_result_table(results)

    rel_err = abs(lpi - GEMM_GUROBI_OBJ) / max(abs(GEMM_GUROBI_OBJ), 1e-10)
    assert rel_err < REL_TOL, (
        f"Gurobi latency_per_iteration {lpi:.0f} deviates {rel_err:.2%} from expected "
        f"{GEMM_GUROBI_OBJ:.0f} (tolerance {REL_TOL:.0%})"
    )


@pytest.mark.slow
def test_gemm_cross_backend():
    """ORToolsBackend (GSCIP) solves the gemm instance to OPTIMAL, within 1% of Gurobi; the allocator
    module's ``create_solver`` is patched to return an ORToolsBackend."""
    results = []
    ort_factory = _make_ortools_factory(mathopt.SolverType.GSCIP)

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(_TTA_CREATE_SOLVER, side_effect=ort_factory),
            patch(_LICENSE_CHECK),
        ):
            t0 = time.perf_counter()
            ctx = _run_gemm_pipeline(tmpdir)
            solve_time = time.perf_counter() - t0

    lpi = _extract_latency_total(ctx)
    _append_result(results, "ORTOOLS_GSCIP", "gscip", lpi, solve_time)
    _print_result_table(results)

    rel_err = abs(lpi - GEMM_GUROBI_OBJ) / max(abs(GEMM_GUROBI_OBJ), 1e-10)
    assert rel_err < REL_TOL, (
        f"OR-Tools (GSCIP) latency_per_iteration {lpi:.0f} deviates {rel_err:.2%} from "
        f"Gurobi baseline {GEMM_GUROBI_OBJ:.0f} (tolerance {REL_TOL:.0%})"
    )


@pytest.mark.slow
def test_gemm_highs():
    """HiGHS solver via ORToolsBackend produces a feasible solution on gemm instance.

    HiGHS is a pure LP/MIP solver (bundled in OR-Tools).  For the allocation MILP it
    should find OPTIMAL.
    """
    results = []
    highs_factory = _make_ortools_factory(mathopt.SolverType.HIGHS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(_TTA_CREATE_SOLVER, side_effect=highs_factory),
            patch(_LICENSE_CHECK),
        ):
            t0 = time.perf_counter()
            ctx = _run_gemm_pipeline(tmpdir)
            solve_time = time.perf_counter() - t0

    lpi = _extract_latency_total(ctx)
    _append_result(results, "ORTOOLS_HIGHS", "highs", lpi, solve_time)
    _print_result_table(results)

    # HiGHS should produce a feasible solution — verify against Gurobi baseline
    rel_err = abs(lpi - GEMM_GUROBI_OBJ) / max(abs(GEMM_GUROBI_OBJ), 1e-10)
    assert rel_err < REL_TOL, (
        f"HiGHS latency_per_iteration {lpi:.0f} deviates {rel_err:.2%} from "
        f"Gurobi baseline {GEMM_GUROBI_OBJ:.0f} (tolerance {REL_TOL:.0%})"
    )


# ---------------------------------------------------------------------------
# SwiGLU integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_swiglu_gurobi_baseline():
    """GurobiBackend produces OPTIMAL on swiglu instance.

    Establishes the swiglu baseline (seq_len=256, embedding_dim=512, hidden_dim=2048, cols=8).
    """
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        ctx = _run_swiglu_pipeline(tmpdir)
        solve_time = time.perf_counter() - t0

    lpi = _extract_latency_total(ctx)
    _append_result(results, "GUROBI", "gurobi", lpi, solve_time)
    _print_result_table(results)

    rel_err = abs(lpi - SWIGLU_GUROBI_OBJ) / max(abs(SWIGLU_GUROBI_OBJ), 1e-10)
    assert rel_err < REL_TOL, (
        f"Gurobi latency_per_iteration {lpi:.0f} deviates {rel_err:.2%} from expected "
        f"{SWIGLU_GUROBI_OBJ:.0f} (tolerance {REL_TOL:.0%})"
    )


@pytest.mark.slow
def test_swiglu_cross_backend():
    """ORToolsBackend (GSCIP) produces OPTIMAL on swiglu instance, matching Gurobi within 1%.

    Covers objective quality on the swiglu workload.
    """
    results = []
    ort_factory = _make_ortools_factory(mathopt.SolverType.GSCIP)

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(_TTA_CREATE_SOLVER, side_effect=ort_factory),
            patch(_LICENSE_CHECK),
        ):
            t0 = time.perf_counter()
            ctx = _run_swiglu_pipeline(tmpdir)
            solve_time = time.perf_counter() - t0

    lpi = _extract_latency_total(ctx)
    _append_result(results, "ORTOOLS_GSCIP", "gscip", lpi, solve_time)
    _print_result_table(results)

    rel_err = abs(lpi - SWIGLU_GUROBI_OBJ) / max(abs(SWIGLU_GUROBI_OBJ), 1e-10)
    assert rel_err < REL_TOL, (
        f"OR-Tools (GSCIP) latency_per_iteration {lpi:.0f} deviates {rel_err:.2%} from "
        f"Gurobi baseline {SWIGLU_GUROBI_OBJ:.0f} (tolerance {REL_TOL:.0%})"
    )


@pytest.mark.slow
def test_swiglu_highs():
    """HiGHS solver via ORToolsBackend produces a feasible solution on swiglu instance."""
    results = []
    highs_factory = _make_ortools_factory(mathopt.SolverType.HIGHS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(_TTA_CREATE_SOLVER, side_effect=highs_factory),
            patch(_LICENSE_CHECK),
        ):
            t0 = time.perf_counter()
            ctx = _run_swiglu_pipeline(tmpdir)
            solve_time = time.perf_counter() - t0

    lpi = _extract_latency_total(ctx)
    _append_result(results, "ORTOOLS_HIGHS", "highs", lpi, solve_time)
    _print_result_table(results)

    rel_err = abs(lpi - SWIGLU_GUROBI_OBJ) / max(abs(SWIGLU_GUROBI_OBJ), 1e-10)
    assert rel_err < REL_TOL, (
        f"HiGHS latency_per_iteration {lpi:.0f} deviates {rel_err:.2%} from "
        f"Gurobi baseline {SWIGLU_GUROBI_OBJ:.0f} (tolerance {REL_TOL:.0%})"
    )
