"""Integration tests for constraint toggle feature (Phase 7).

TEST-01: Infeasibility-flip tests — each constraint group's guard is structurally
effective (tight limit + enabled = RuntimeError, tight limit + disabled = success).
TEST-02: Cross-backend parity — Gurobi and OR-Tools agree within tolerance with
selective constraints active.
"""

import os
import re
import tempfile
from unittest.mock import patch

import pytest
from ortools.math_opt.python import mathopt

from stream.api import optimize_allocation_co
from stream.hardware.architecture.core import Core
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.opt.solver import ConstraintSelection, ORToolsBackend

# ---------------------------------------------------------------------------
# Constants (same as test_cross_backend.py)
# ---------------------------------------------------------------------------
ACCELERATOR = os.path.join(
    os.path.dirname(__file__),
    "../../stream/inputs/aie/hardware/whole_array_strix.yaml",
)
REL_TOL = 0.01

_ALLOC_CREATE_SOLVER = "stream.opt.allocation.constraint_optimization.allocation.create_solver"
_TTA_CREATE_SOLVER = "stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation.create_solver"
_LICENSE_CHECK = "stream.api._sanity_check_gurobi_license"
_BUILD_TRANSFER_CONTEXT = (
    "stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation.build_transfer_context"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from stream.opt.allocation.constraint_optimization.context import (  # noqa: E402
    build_transfer_context as _real_build_transfer_context,
)


def _run_gemm(output_path: str, constraint_selection: ConstraintSelection | None = None):
    """Run the TETRA GEMM pipeline with optional constraint selection."""
    M, K, N = 256, 8192, 2048
    m, k, n = 32, 32, 32
    in_dtype, out_dtype = "bf16", "bf16"
    nb_rows, nb_cols = 4, 8

    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    mapping_path = make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use=nb_rows, nb_cols_to_use=nb_cols)

    hw_name = ACCELERATOR.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}-{nb_rows}_row_{nb_cols}_col"

    return optimize_allocation_co(
        hardware=ACCELERATOR,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path=output_path,
        skip_if_exists=False,
        nb_cols_to_use=nb_cols,
        constraint_selection=constraint_selection,
    )


def _make_ortools_factory(solver_type: mathopt.SolverType = mathopt.SolverType.GSCIP):
    """Return a drop-in replacement for ``create_solver`` that always returns an ``ORToolsBackend``."""

    def _factory(backend, name="", *, solver_type=solver_type, **kwargs):  # noqa: ARG001
        return ORToolsBackend(name, solver_type)

    return _factory


def _extract_latency_total(ctx) -> float:
    """Extract ``latency_total`` (solver objective) from a completed pipeline context."""
    scheduler = ctx.get("scheduler")
    return float(scheduler.latency_total)


def _build_transfer_context_tight_dma(*args, **kwargs):
    """Wrap build_transfer_context with DMA channels set to 1 (infeasibly tight)."""
    kwargs["max_compute_tile_dma_channels"] = 1
    kwargs["max_mem_tile_dma_channels"] = 1
    kwargs["max_shim_tile_dma_channels"] = 1
    return _real_build_transfer_context(*args, **kwargs)


# ---------------------------------------------------------------------------
# TEST-01: Infeasibility-flip tests — one per constraint group
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_memory_capacity_flip():
    """memory_capacity guard: tight limit (1 bit) causes infeasibility; disabling restores feasibility.

    D-01: The guard in _create_constraints() is structurally wired. Proof:
    - tight limit + memory_capacity=True  -> RuntimeError (solver infeasible)
    - tight limit + memory_capacity=False -> success (constraint skipped)
    """
    cs_all_off = ConstraintSelection(
        memory_capacity=False,
        object_fifo_depth=False,
        buffer_descriptors=False,
        dma_channels=False,
    )

    # Enabled + tight limit -> infeasible
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "get_memory_capacity", return_value=1):
            with pytest.raises(RuntimeError):
                _run_gemm(
                    tmpdir,
                    constraint_selection=ConstraintSelection(
                        memory_capacity=True,
                        object_fifo_depth=False,
                        buffer_descriptors=False,
                        dma_channels=False,
                    ),
                )

    # Disabled + tight limit -> feasible (constraint skipped entirely)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "get_memory_capacity", return_value=1):
            ctx = _run_gemm(tmpdir, constraint_selection=cs_all_off)
    assert _extract_latency_total(ctx) > 0


@pytest.mark.slow
def test_object_fifo_depth_flip():
    """object_fifo_depth guard: tight FIFO limit causes infeasibility; disabling restores feasibility.

    D-01: The guard in _create_constraints() is structurally wired. Proof:
    - tight max_object_fifo_depth=1 + object_fifo_depth=True  -> RuntimeError
    - tight max_object_fifo_depth=1 + object_fifo_depth=False -> success

    Since Core.__init__ explicitly sets self.max_object_fifo_depth from constructor
    args, a class-level attribute patch is shadowed by instance attributes. We wrap
    __init__ to override the value after construction.
    """
    _original_init = Core.__init__

    def _tight_fifo_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.max_object_fifo_depth = 1

    # Enabled + tight limit -> infeasible
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            with pytest.raises(RuntimeError):
                _run_gemm(
                    tmpdir,
                    constraint_selection=ConstraintSelection(
                        memory_capacity=False,
                        object_fifo_depth=True,
                        buffer_descriptors=False,
                        dma_channels=False,
                    ),
                )

    # Disabled + tight limit -> feasible (constraint skipped entirely)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            ctx = _run_gemm(
                tmpdir,
                constraint_selection=ConstraintSelection(
                    memory_capacity=False,
                    object_fifo_depth=False,
                    buffer_descriptors=False,
                    dma_channels=False,
                ),
            )
    assert _extract_latency_total(ctx) > 0


@pytest.mark.slow
def test_buffer_descriptor_flip():
    """buffer_descriptors guard: tight BD limit causes infeasibility; disabling restores feasibility.

    D-01: The guard in _create_constraints() is structurally wired. Proof:
    - tight max_object_fifo_depth=1 + buffer_descriptors=True  -> RuntimeError
    - tight max_object_fifo_depth=1 + buffer_descriptors=False -> success

    Note: BD constraints share max_object_fifo_depth as the RHS (per research pitfall 6).
    The toggle is still isolated by the ConstraintSelection.buffer_descriptors field.
    We disable object_fifo_depth in both arms to isolate the BD constraint.
    """
    _original_init = Core.__init__

    def _tight_fifo_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.max_object_fifo_depth = 1

    # Enabled + tight limit -> infeasible
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            with pytest.raises(RuntimeError):
                _run_gemm(
                    tmpdir,
                    constraint_selection=ConstraintSelection(
                        memory_capacity=False,
                        object_fifo_depth=False,
                        buffer_descriptors=True,
                        dma_channels=False,
                    ),
                )

    # Disabled + tight limit -> feasible (constraint skipped entirely)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            ctx = _run_gemm(
                tmpdir,
                constraint_selection=ConstraintSelection(
                    memory_capacity=False,
                    object_fifo_depth=False,
                    buffer_descriptors=False,
                    dma_channels=False,
                ),
            )
    assert _extract_latency_total(ctx) > 0


@pytest.mark.slow
def test_dma_channels_flip():
    """dma_channels guard: tight DMA limit causes infeasibility; disabling restores feasibility.

    D-01: The guard in _overlap_and_objective() is structurally wired. Proof:
    - DMA channels=1 (all tiles) + dma_channels=True  -> RuntimeError
    - DMA channels=1 (all tiles) + dma_channels=False -> success

    Patch target: build_transfer_context in TTA's own namespace (imported at line 22).
    """
    # Enabled + tight limit -> infeasible
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(_BUILD_TRANSFER_CONTEXT, side_effect=_build_transfer_context_tight_dma):
            with pytest.raises(RuntimeError):
                _run_gemm(
                    tmpdir,
                    constraint_selection=ConstraintSelection(
                        memory_capacity=False,
                        object_fifo_depth=False,
                        buffer_descriptors=False,
                        dma_channels=True,
                    ),
                )

    # Disabled + tight limit -> feasible (DMA constraint and objective terms skipped)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(_BUILD_TRANSFER_CONTEXT, side_effect=_build_transfer_context_tight_dma):
            ctx = _run_gemm(
                tmpdir,
                constraint_selection=ConstraintSelection(
                    memory_capacity=False,
                    object_fifo_depth=False,
                    buffer_descriptors=False,
                    dma_channels=False,
                ),
            )
    assert _extract_latency_total(ctx) > 0


# ---------------------------------------------------------------------------
# TEST-02: Cross-backend parity with selective constraints (Gurobi vs OR-Tools)
# ---------------------------------------------------------------------------

_PARITY_CASES = [
    pytest.param(
        ConstraintSelection(memory_capacity=False, object_fifo_depth=False),
        id="memory_off",
        # NOTE: Disables both memory_capacity AND object_fifo_depth per SEL-05
        # nonsensical-combination rule. Memory capacity constraints use FIFO depth
        # as RHS, so disabling memory alone (with FIFO still active) would leave
        # a misleading constraint configuration. This is intentional -- "memory off"
        # means both fields are disabled as a semantic unit.
    ),
    pytest.param(
        ConstraintSelection(object_fifo_depth=False),
        id="fifo_off",
    ),
    pytest.param(
        ConstraintSelection(buffer_descriptors=False),
        id="bd_off",
    ),
    pytest.param(
        ConstraintSelection(dma_channels=False),
        id="dma_off",
    ),
    pytest.param(
        ConstraintSelection(memory_capacity=False, object_fifo_depth=False, dma_channels=False),
        id="memory_and_dma_off",
    ),
    pytest.param(
        ConstraintSelection(object_fifo_depth=False, buffer_descriptors=False),
        id="fifo_and_bd_off",
    ),
    pytest.param(
        ConstraintSelection(
            memory_capacity=False, object_fifo_depth=False, buffer_descriptors=False, dma_channels=False
        ),
        id="all_off",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("cs", _PARITY_CASES)
def test_cross_backend_parity(cs: ConstraintSelection):
    """Gurobi and OR-Tools agree within REL_TOL for constraint selection *cs*.

    Per D-03: 7 combinations tested (4 individual toggles + 3 multi-toggle combos).
    Per research Pitfall 3: dynamic Gurobi reference (not hardcoded baseline) because
    disabling DMA changes the objective formulation (no DMA penalty terms).
    """
    # 1. Run Gurobi (unpatched) as dynamic reference
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx_gurobi = _run_gemm(tmpdir, constraint_selection=cs)
    gurobi_obj = _extract_latency_total(ctx_gurobi)

    # 2. Run OR-Tools (patched) with same constraint selection
    ort_factory = _make_ortools_factory()
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(_ALLOC_CREATE_SOLVER, side_effect=ort_factory),
            patch(_TTA_CREATE_SOLVER, side_effect=ort_factory),
            patch(_LICENSE_CHECK),
        ):
            ctx_ort = _run_gemm(tmpdir, constraint_selection=cs)
    ort_obj = _extract_latency_total(ctx_ort)

    # 3. Assert parity within tolerance
    rel_err = abs(ort_obj - gurobi_obj) / max(abs(gurobi_obj), 1e-10)
    assert rel_err < REL_TOL, (
        f"OR-Tools objective {ort_obj:.0f} deviates {rel_err:.2%} from "
        f"Gurobi {gurobi_obj:.0f} (tolerance {REL_TOL:.0%}, cs={cs})"
    )
