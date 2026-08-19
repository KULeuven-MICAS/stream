"""Regression: the AIE Strix auto-map must not crash reconciling a compute split with memory cores."""

from __future__ import annotations

from pathlib import Path

import pytest

from stream.api import optimize_allocation_co_generic
from stream.ir.infeasibility import InfeasibleAllocationError

_ACCELERATOR = "stream/inputs/aie/hardware/whole_array_strix.yaml"
_ATTENTION = "stream/inputs/testing/workload/attention_head.onnx"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_attention_strix_no_tiling_crash(tmp_path: Path) -> None:
    try:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=_ATTENTION,
            experiment_id="attention_strix_regression",
            output_path=str(tmp_path),
            nb_cols_to_use=4,
            backend="ortools_gscip",
        )
    except InfeasibleAllocationError:
        # A structured infeasibility is a well-defined outcome, not the crash under test.
        return
    assert ctx is not None
