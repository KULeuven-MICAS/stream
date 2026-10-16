"""Tests for the canonical architecture builders (stream.workload.models).

These pin the *internal representation* the framework showcases: attention is affine MatMuls with
a single Softmax barrier; Mamba is a SEQUENTIAL recurrence that chunks. They are representation
tests, not numerical ones (the chunked-scan math is verified in tests/rewrites).
"""

from __future__ import annotations

import pytest

from stream.api import evaluate_mapping
from stream.workload.blocks import build_block
from stream.workload.iterator_type import IteratorType, derive_iterator_types, is_state_operand, sequential_dims
from stream.workload.models import MODEL_CATALOG, build_attention_block, build_mamba_block
from stream.workload.node import FusionEdge, NormalizationNode
from stream.workload.normalization import parallel_axes, reduction_axes


def test_catalog_builds_every_model():
    keys = {s.key for s in MODEL_CATALOG}
    assert {"attention", "gqa", "linear_attention", "mamba", "kv_cache"} <= keys
    for spec in MODEL_CATALOG:
        wl = spec.build()
        assert wl.get_computation_nodes(), f"{spec.key} produced no compute nodes"


def test_attention_softmax_is_a_fusible_normalization_not_a_barrier():
    wl = build_attention_block()
    assert [n for n in wl.nodes if isinstance(n, FusionEdge)] == []  # no barriers
    softmax = next(n for n in wl.get_computation_nodes() if isinstance(n, NormalizationNode))
    # reduces the key axis (position 3), parallel over batch/head/query
    assert reduction_axes(softmax) == (3,)
    assert parallel_axes(softmax) == (0, 1, 2)
    # the affine MatMuls each contract exactly the expected axes; none is recurrent
    for n in wl.get_computation_nodes():
        if isinstance(n, NormalizationNode):
            continue
        assert n.type == "MatMul"
        assert any(t == IteratorType.REDUCTION for t in derive_iterator_types(n).values())
        assert sequential_dims(n) == frozenset()


def test_attention_is_one_fusible_region():
    """Softmax is parallel over b,h,i, so the whole block is one fusible region (flash-attention view)."""
    wl = build_attention_block()
    groups = wl.split_fusion_groups()
    assert len(groups) == 1
    names = {c.name for c in groups[0].get_computation_nodes()}
    assert {"scores", "softmax", "context"} <= names
    assert len(groups[0].get_dimension_sizes()) > 0


def test_mamba_state_update_decomposes_to_paper_subops():
    """The Mamba block decomposes into the paper's Fig. 7 state-update affine sub-ops."""
    wl = build_mamba_block()
    by_name = {n.name: n for n in wl.get_computation_nodes()}
    assert set(by_name) == {"dA", "Abar", "dB", "dBx", "scan", "readout", "skip", "out"}
    assert by_name["Abar"].type == "Exp"  # the discretized decay exp(dA), a multi-cycle op


def test_mamba_scan_is_sequential_over_the_token_axis():
    """The selective scan reads state at t-1: the token axis is SEQUENTIAL, channel/state PARALLEL."""
    scan = next(n for n in build_mamba_block().get_computation_nodes() if n.type == "SelectiveScan")
    assert sequential_dims(scan) == frozenset({0})  # the token axis carries the state
    types = derive_iterator_types(scan)
    assert types[0] == IteratorType.SEQUENTIAL
    assert types[1] == IteratorType.PARALLEL and types[2] == IteratorType.PARALLEL


def test_mamba_readout_reduces_the_state_axis():
    """y'_t = sum_N C_t · h_t contracts the state dimension N (a linear reduction, not a barrier)."""
    readout = next(n for n in build_mamba_block().get_computation_nodes() if n.name == "readout")
    types = derive_iterator_types(readout)
    assert [p for p, t in types.items() if t == IteratorType.REDUCTION] == [2]  # exactly the state axis N


def test_mamba_fuses_into_one_region():
    """No data-dependent read and no nonlinear reduction: the state-update block is one fusible region."""
    groups = build_mamba_block().split_fusion_groups()
    assert len(groups) == 1
    names = {c.name for c in groups[0].get_computation_nodes()}
    assert {"dA", "Abar", "scan", "readout"} <= names


@pytest.mark.parametrize(
    ("block", "config"),
    [("mamba", {"seq": 16, "d_inner": 16, "d_state": 4}), ("linear_attention", {"seq": 8, "d_k": 8, "d_v": 8})],
)
def test_a_recurrence_with_an_initial_state_input_solves_end_to_end(tmp_path, block, config):
    """The carried state stays resident on its node: no transfer moves it and its input edge feeds nothing."""
    workload = build_block(block, **config)
    states = {t.name for n in workload.get_computation_nodes() for t in n.inputs if is_state_operand(n, t)}
    assert states
    allocation = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml", workload, str(tmp_path)
    ).context.get("allocation")
    assert allocation.solution.latency.total > 0
    lowered = allocation.problem.workload
    assert not {t.name for tr in lowered.get_transfer_nodes() for t in tr.inputs} & states
    assert not {t.name for e in lowered.get_in_edges() for t in e.outputs} & states


def test_self_attention_keeps_its_query_and_key_axes_apart_in_the_steady_state(tmp_path):
    """The queries and the keys both read the input rows: each reader gets its own copy, so the scores still span
    every query by every key."""
    workload = build_block("attention", batch=1, heads=2, seq=16, d_head=16)
    allocation = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml", workload, str(tmp_path)
    ).context.get("allocation")
    lowered = allocation.problem.workload
    scores = next(n for n in lowered.get_computation_nodes() if n.name == "scores")
    query, key = (lowered.get_dims(scores)[p] for p in (2, 3))
    assert query != key
    assert {tr.name for tr in lowered.get_transfer_nodes() if tr.inputs[0].name == "x"} == {
        "Transfer(x)",
        "Transfer(x for proj_q)",
    }
