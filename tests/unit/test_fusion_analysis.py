"""Tests for the affine fusion analysis (M06): fusibility + streaming buffer from map composition.

The proof points (plan/06): a recurrence streams along its sequence axis with an O(1) buffer, a
conv-style window grows a line buffer + halo, and a full-reduction (softmax) must hold the whole
reduced axis — reported, never silently assumed streamable.
"""

from __future__ import annotations

from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.workload.fusion.analysis import pairwise_fusion, shared_tensor
from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor


def _node(name: str, ins, outs, maps, op: str = "Op") -> ComputationNode:
    return ComputationNode(type=op, name=name, inputs=ins, outputs=outs, operand_mapping=maps)


def _identity2() -> AffineMap:
    return AffineMap.from_callable(lambda a, b: (a, b))


def test_recurrence_streams_with_o1_buffer():
    """A t-1 state carry: window 1, advancing by 1, buffer independent of the sequence length."""

    def build(seq_len: int):
        h = Tensor.create("h", bf16, (seq_len, 8))
        x = Tensor.create("x", bf16, (seq_len, 8))
        prod = _node("prod", (x,), (h,), (_identity2(), _identity2()))
        out = Tensor.create("out", bf16, (seq_len, 8))
        cons = _node("cons", (h,), (out,), (AffineMap.from_callable(lambda t, d: (t - 1, d)), _identity2()))
        return pairwise_fusion(prod, cons, fusion_dim=0, extents={0: seq_len, 1: 8})

    short, long = build(16), build(128)
    assert short.window == 1 and short.step == 1 and short.fusible
    assert short.buffer_elements == long.buffer_elements == 8  # O(1) in L — the recurrence proof point


def test_conv_window_grows_a_line_buffer_with_halo():
    """A 3-wide window read: window 3, step 1, halo 2, buffer = window × the other-axis extent."""
    x = Tensor.create("x", bf16, (16, 8))
    s = Tensor.create("s", bf16, (16, 8))
    prod = _node("prod", (x,), (s,), (_identity2(), _identity2()))
    o = Tensor.create("o", bf16, (14, 8))
    cons = _node(
        "cons",
        (s,),
        (o,),
        (AffineMap.from_callable(lambda ot, d, k: (ot + k, d)), AffineMap.from_callable(lambda ot, d, k: (ot, d))),
    )
    w = pairwise_fusion(prod, cons, fusion_dim=0, extents={0: 14, 1: 8, 2: 3})
    assert w.window == 3 and w.step == 1 and w.halo == 2 and w.fusible
    assert w.buffer_elements == 3 * 8


def test_softmax_reduction_holds_the_whole_row():
    """Fusing a full reduction (a softmax stat) forces holding the entire reduced axis — the buffer is
    the full row, not a small line buffer. The analysis must surface this, not assume streamability."""
    x = Tensor.create("x", bf16, (12, 12))
    scores = Tensor.create("scores", bf16, (12, 12))
    prod = _node("scores", (x,), (scores,), (_identity2(), _identity2()))
    stat = Tensor.create("stat", bf16, (12,))
    reduce_node = _node(
        "reduce",
        (scores,),
        (stat,),
        (_identity2(), AffineMap.from_callable(lambda i, j: (i,))),
        op="ReduceMax",
    )
    w = pairwise_fusion(prod, reduce_node, fusion_dim=0, extents={0: 12, 1: 12})
    # Per output row the reduction reads the entire j axis -> the buffer spans the full row (12),
    # not a bounded window: this is what makes softmax non-streamable along the reduced axis.
    assert w.buffer_elements == 12


def test_shared_tensor_detection():
    a = Tensor.create("a", bf16, (4, 4))
    t = Tensor.create("t", bf16, (4, 4))
    b = Tensor.create("b", bf16, (4, 4))
    prod = _node("p", (a,), (t,), (_identity2(), _identity2()))
    cons = _node("c", (t,), (b,), (_identity2(), _identity2()))
    assert shared_tensor(prod, cons) is t
