"""Regression: the AIE Strix auto-map must not crash reconciling a compute split with memory cores.

The generic auto-mapper split ``attention``'s scores matmul 8 ways onto cores spanning 3 array
columns, so the steady-state scheduler asked for 3 memory tiles to stage a transfer whose compute
split was 8. ``get_matching_tiling`` required the memory-core count to divide the split (3 does not
divide 8) and raised ``ValueError`` before the MILP ever solved. The count is now snapped to a
divisor of the split, so the run reaches the solver and returns a well-defined result.

The outcome (feasible or infeasible) is not pinned -- only that no reconciliation error escapes the
scheduler. The crash was pre-solve, so this holds without a Gurobi license (OR-Tools SCIP).
"""

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
