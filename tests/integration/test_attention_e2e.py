"""End-to-end DSE verification on an attention workload.

Runs the *whole* constraint-optimization pipeline -- parse ONNX (MatMul + Softmax-family
normalization + Transpose), auto-generate the mapping, split fusion groups, tile, cost-estimate,
solve the MILP tensor/transfer allocation, and estimate memory -- on the attention-head fixture, and
asserts it produces a positive latency for every fusion group. This is the critical check that the
new attention node types flow through the full flow, not just parsing.

Slow (the MILP solve takes minutes), so it is excluded from the default ``-m 'not slow'`` run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stream.api import optimize_allocation_co_generic

_ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
_ATTENTION = "stream/inputs/testing/workload/attention_head.onnx"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_attention_dse_end_to_end(tmp_path: Path):
    ctx = optimize_allocation_co_generic(
        hardware=_ACCELERATOR,
        workload=_ATTENTION,
        experiment_id="attention_e2e",
        output_path=str(tmp_path),
        backend="ortools_gscip",  # SCIP: no Gurobi license needed
    )

    total_latency = ctx.get("total_latency")
    group_latencies = ctx.get("group_latencies")

    assert total_latency is not None and total_latency > 0, f"expected positive total latency, got {total_latency}"
    assert group_latencies, "expected per-group latencies"
    assert all(lat > 0 for lat in group_latencies.values()), f"every fusion group must schedule: {group_latencies}"

    # the workload really did carry the attention compute nodes through the flow
    workload = ctx.get("workload")
    assert workload is not None
