"""Dimension-structure and dataflow tests for attention variants: GQA/MQA and linear attention.

These check the *representation* — which axes are parallel/reduction/sequential and which operand is
reused/carried — and the *dataflow* between back-to-back operators (what data a tile needs before it
can compute), using the derived affine machinery.
"""

from __future__ import annotations

import pytest

from stream.workload.affine_access import footprint, relevancy
from stream.workload.iterator_type import (
    IteratorType,
    SequentialUnrollError,
    check_spatial_unroll_legal,
    derive_iterator_types,
    is_state_operand,
    sequential_dims,
)
from stream.workload.models import (
    GQAConfig,
    build_attention_block,
    build_gqa_block,
    build_linear_attention_block,
)
from stream.workload.steady_state.iteration_space import LoopEffect

# --------------------------------------------------------------------------- GQA / MQA


def _node(wl, name):
    return next(n for n in wl.get_computation_nodes() if n.name == name)


def test_gqa_kv_is_reused_across_query_heads():
    """The rep axis r (position 2 in b,g,r,i,j,e) is INVARIANT for K and V but VARYING for Q --
    every query head in a group reuses the same K/V. That reuse *is* GQA's data saving."""
    wl = build_gqa_block()
    scores = _node(wl, "scores")
    q, k = scores.inputs
    rep_axis = 2
    assert relevancy(scores, k, rep_axis) == LoopEffect.INVARIANT
    assert relevancy(scores, q, rep_axis) == LoopEffect.VARYING
    context = _node(wl, "context")
    v = context.inputs[1]
    assert relevancy(context, v, rep_axis) == LoopEffect.INVARIANT


def test_gqa_contraction_axes():
    """scores reduce the head dim e; context reduces the key position j."""
    wl = build_gqa_block()
    scores_red = [p for p, t in derive_iterator_types(_node(wl, "scores")).items() if t == IteratorType.REDUCTION]
    ctx_red = [p for p, t in derive_iterator_types(_node(wl, "context")).items() if t == IteratorType.REDUCTION]
    assert scores_red == [5]  # e
    assert ctx_red == [5]  # j (last iteration dim of the context node)


def test_mqa_is_single_group_shared_kv():
    """MQA = one KV group: still valid, K/V reused across every query head (invariant rep axis)."""
    wl = build_gqa_block(GQAConfig(groups=1, reps=8))
    scores = _node(wl, "scores")
    assert relevancy(scores, scores.inputs[1], 2) == LoopEffect.INVARIANT
    # the group axis exists but has extent 1
    assert scores.outputs[0].shape[1] == 1


# --------------------------------------------------------------------------- linear attention


def test_linear_attention_is_a_recurrence():
    """The state read S[t-1] makes t SEQUENTIAL; the state input is detected as the carry."""
    wl = build_linear_attention_block()
    upd = _node(wl, "state_update")
    assert sequential_dims(upd) == frozenset({0})
    state_prev = next(o for o in upd.inputs if o.name == "state_prev")
    assert is_state_operand(upd, state_prev)
    assert derive_iterator_types(upd)[0] == IteratorType.SEQUENTIAL


def test_linear_attention_sequential_axis_cannot_be_unrolled():
    """A recurrence's sequence axis keeps a total order — spatial unrolling must be rejected."""
    upd = _node(build_linear_attention_block(), "state_update")
    with pytest.raises(SequentialUnrollError):
        check_spatial_unroll_legal(upd, [0])  # t
    check_spatial_unroll_legal(upd, [1])  # d_k is fine


def test_linear_attention_readout_contracts_dk():
    read = _node(build_linear_attention_block(), "readout")
    red = [p for p, t in derive_iterator_types(read).items() if t == IteratorType.REDUCTION]
    assert red == [2]  # d_k


def test_recurrence_tile_depends_on_previous_state():
    """Dataflow: computing state at t needs the state at t-1 (the O(1) carry distance)."""
    upd = _node(build_linear_attention_block(), "state_update")
    state_prev = next(o for o in upd.inputs if o.name == "state_prev")
    prev_map = upd.get_mapping(state_prev)
    # a tile at t=5 (dk,dv full) reads state_prev at t=4
    region = footprint(prev_map, {0: range(5, 6), 1: range(0, 32), 2: range(0, 32)})
    assert region[0] == range(4, 5)


# --------------------------------------------------------------------------- back-to-back dataflow


def test_context_needs_the_whole_attention_row():
    """Before the context matmul can produce O[b,h,i,e] it needs the entire softmax row P[b,h,i,:]
    (j is the reduction) — the data-dependency that makes softmax a reduction the context waits on."""
    wl = build_attention_block()
    context = _node(wl, "context")
    probs = context.inputs[0]
    p_map = context.get_mapping(probs)  # (b,h,i,e,j) -> (b,h,i,j)
    seq = probs.shape[-1]
    # one output element (b,h,i,e fixed) iterating the full key axis j
    tile = {0: range(0, 1), 1: range(0, 1), 2: range(0, 1), 3: range(0, 1), 4: range(0, seq)}
    region = footprint(p_map, tile)
    assert region[-1] == range(0, seq)  # the whole row of keys
