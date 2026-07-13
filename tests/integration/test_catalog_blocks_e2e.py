"""End-to-end allocation of the affine-IR model-catalog blocks (MHA / linear attention).

Runs the whole constraint-optimization pipeline on an *in-memory* ``Workload`` (no ONNX round-trip)
via ``optimize_allocation_co_generic_workload``. The softmax is a nonlinear reduction, not a hard
barrier, so the whole attention chain fuses into ONE region (its reduction axis kept resident, never
spilled); a SEQUENTIAL recurrence (linear attention) likewise stays a single group. Slow (the MILP
solve takes a while), so excluded from the default ``-m 'not slow'`` run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stream.api import optimize_allocation_co_generic_workload
from stream.workload.models import (
    AttentionConfig,
    LinearAttentionConfig,
    build_attention_block,
    build_linear_attention_block,
)

_ACCELERATOR = "stream/inputs/testing/hardware/tpu_like_quad_core.yaml"


def _run(workload, tmp_path: Path):
    return optimize_allocation_co_generic_workload(
        hardware=_ACCELERATOR,
        workload=workload,
        experiment_id="catalog_e2e",
        output_path=str(tmp_path),
        backend="ORTOOLS_GSCIP",  # SCIP: no Gurobi license needed
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_attention_block_fuses_into_one_region(tmp_path: Path):
    ctx = _run(build_attention_block(AttentionConfig(batch=1, heads=1, seq=8, d_head=8)), tmp_path)

    group_latencies = ctx.get("group_latencies")
    assert group_latencies is not None and len(group_latencies) == 1, (
        f"the softmax is not a hard barrier -- attention must fuse into one region, got {group_latencies}"
    )
    assert all(lat > 0 for lat in group_latencies.values()), f"the fused region must schedule: {group_latencies}"
    assert ctx.get("total_latency") == pytest.approx(sum(group_latencies.values()))


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_softmax_is_decomposed_in_the_pipeline(tmp_path: Path):
    """The generic pipeline decomposes the MHA softmax into ReduceMax/Exp/ReduceSum/Div and the whole
    block still solves feasibly -- so the two reduction passes are cost-modelled end to end."""
    ctx = optimize_allocation_co_generic_workload(
        hardware=_ACCELERATOR,
        workload=build_attention_block(AttentionConfig(batch=1, heads=1, seq=8, d_head=8)),
        experiment_id="expanded_softmax",
        output_path=str(tmp_path),
        backend="ORTOOLS_GSCIP",
    )
    assert ctx.get("total_latency") and ctx.get("total_latency") > 0
    node_types = {n.type for n in ctx.get("workload").get_computation_nodes()}
    assert {"ReduceMax", "Exp", "ReduceSum", "Div"} <= node_types
    assert "Softmax" not in node_types


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_linear_attention_recurrence_is_a_single_group(tmp_path: Path):
    ctx = _run(build_linear_attention_block(LinearAttentionConfig(seq=8, d_k=8, d_v=8)), tmp_path)

    group_latencies = ctx.get("group_latencies")
    assert group_latencies is not None and len(group_latencies) == 1, (
        f"the SEQUENTIAL recurrence must stay one group, got {group_latencies}"
    )
    assert ctx.get("total_latency") > 0
