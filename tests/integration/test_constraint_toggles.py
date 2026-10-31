"""Integration tests for switching constraint families off.

Flip tests: each constraint family is structurally effective, a tight limit binding when the family is
selected and not when it is left out.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from stream.api import SolveOptions, default_families, evaluate_mapping
from stream.hardware.architecture.core import Core
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.opt.allocation.constraint_optimization.families import FamilySpec
from stream.opt.allocation.constraint_optimization.space import DecisionSpace

# ---------------------------------------------------------------------------
# Constants (same as test_cross_backend.py)
# ---------------------------------------------------------------------------
ACCELERATOR = os.path.join(
    os.path.dirname(__file__),
    "../../stream/inputs/aie/hardware/whole_array_strix.yaml",
)

_GROUPS = ("memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels")
_TIGHT_DMA = {
    "aie2_dma_channels": {
        "max_compute_tile_dma_channels": 1,
        "max_mem_tile_dma_channels": 1,
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _without(*off: str) -> tuple[FamilySpec, ...]:
    """The default families with the constraint groups ``off`` switched off as 1.x's toggles did: the object-FIFO
    depth's family keeps its buffering level, every other group's family is left out."""
    depth = {"object_fifo_depth": {"depth": False}} if "object_fifo_depth" in off else None
    return default_families(ACCELERATOR, [g for g in off if g != "object_fifo_depth"], depth)


def _only(*kept: str) -> tuple[FamilySpec, ...]:
    """The default families with, of the four toggled constraint groups, only those ``kept`` on."""
    return _without(*(g for g in _GROUPS if g not in kept))


def _run_gemm(output_path: str, families: tuple[FamilySpec, ...] | None = None):
    """Run the GEMM pipeline with the given constraint families."""
    M, K, N = 256, 8192, 2048
    m, k, n = 32, 32, 32
    in_dtype, out_dtype = "bf16", "bf16"
    nb_rows, nb_cols = 4, 8

    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    mapping_path = make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use=nb_rows, nb_cols_to_use=nb_cols)

    return evaluate_mapping(
        ACCELERATOR,
        workload_path,
        output_path,
        mapping_path,
        options=SolveOptions(nb_cols_to_use=nb_cols, families=families),
    ).context


def _extract_latency_total(ctx) -> float:
    """Extract ``latency_total`` (solver objective) from a completed pipeline context."""
    return float(ctx.get("allocation").solution.latency.total)


# ---------------------------------------------------------------------------
# Flip tests, one per constraint group
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_memory_capacity_flip():
    """memory_capacity family: every memory at the least its pinned tensors need, so the capacity screen, which runs
    whatever the families, passes, but holding more costs a slower allocation; leaving the family out lifts the limit.
    """
    least = 64 * 8192  # bits: the pinned tensors of compute tile 0, the most any memory holds

    shrunk = patch.object(DecisionSpace, "memory_capacity_bits", return_value=least)
    with tempfile.TemporaryDirectory() as tmpdir, shrunk:
        limited = _extract_latency_total(_run_gemm(tmpdir, families=_only("memory_capacity")))
    with tempfile.TemporaryDirectory() as tmpdir, shrunk:
        unlimited = _extract_latency_total(_run_gemm(tmpdir, families=_only()))
    assert unlimited < limited


@pytest.mark.slow
def test_object_fifo_depth_flip():
    """object_fifo_depth family: tight FIFO limit causes infeasibility; leaving it out restores feasibility.

    - tight max_object_fifo_depth=1 + object_fifo_depth selected  -> RuntimeError
    - tight max_object_fifo_depth=1 + object_fifo_depth left out -> success

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
                _run_gemm(tmpdir, families=_only("object_fifo_depth"))

    # Disabled + tight limit -> feasible (constraint skipped entirely)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            ctx = _run_gemm(tmpdir, families=_only())
    assert _extract_latency_total(ctx) > 0


@pytest.mark.slow
def test_buffer_descriptor_flip():
    """buffer_descriptors family: tight BD limit causes infeasibility; leaving it out restores feasibility.

    - tight max_object_fifo_depth=1 + buffer_descriptors selected  -> RuntimeError
    - tight max_object_fifo_depth=1 + buffer_descriptors left out -> success

    Note: BD constraints share max_object_fifo_depth as the RHS, so object_fifo_depth
    is left out in both arms to isolate the BD constraint.
    """
    _original_init = Core.__init__

    def _tight_fifo_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.max_object_fifo_depth = 1

    # Enabled + tight limit -> infeasible
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            with pytest.raises(RuntimeError):
                _run_gemm(tmpdir, families=_only("buffer_descriptors"))

    # Disabled + tight limit -> feasible (constraint skipped entirely)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(Core, "__init__", _tight_fifo_init):
            ctx = _run_gemm(tmpdir, families=_only())
    assert _extract_latency_total(ctx) > 0


@pytest.mark.slow
def test_dma_channels_flip():
    """A tight DMA limit (one channel per tile, the aie2_dma_channels options) is infeasible with dma_channels
    selected and feasible with it left out, which leaves out the limit too."""
    tight = tuple(_TIGHT_DMA if family == "aie2_dma_channels" else family for family in _only("dma_channels"))
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError):
            _run_gemm(tmpdir, families=tight)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = _run_gemm(tmpdir, families=_only())
    assert _extract_latency_total(ctx) > 0
