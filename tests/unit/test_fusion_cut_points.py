"""The affine fusion analysis drives where the generic pipeline splits fusion groups.

A nonlinear reduction (Softmax/LayerNorm) and a data-dependent read (MoE dispatch/combine, gather)
are barriers that cannot be streamed, so they end a fusion group. A linear contraction (a MatMul's
reduction axis) and a SEQUENTIAL recurrence are *not* barriers -- they stay inside one group. CNN
graphs carry no such barriers, so their cut points are unchanged.
"""

from __future__ import annotations

from functools import cache

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.blocks import build_block
from stream.workload.fusion.analysis import barrier_cut_points
from stream.workload.models import (
    build_attention_block,
    build_gqa_block,
    build_kv_cache_decode_step,
    build_linear_attention_block,
    build_mamba_block,
)
from stream.workload.workload import determine_fusion_cut_points

CONV_FIXTURE = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"


@cache
def _conv_workload():
    parser = ONNXModelParser(CONV_FIXTURE)
    parser.run()
    return parser.workload


def test_softmax_is_a_cut_point_in_attention():
    """MHA/GQA/KV-cache decode: the softmax nonlinear reduction ends its group (probs materialized),
    so the block splits into {..., scores, softmax} | {context, ...}."""
    for build in (build_attention_block, build_gqa_block, build_kv_cache_decode_step):
        assert determine_fusion_cut_points(build()) == ["softmax"]


def test_data_dependent_combine_is_a_cut_point_in_moe():
    """The MoE combine reads its input via a runtime routing table -- a hard barrier -- so the
    producing expert GEMM ends its group."""
    assert determine_fusion_cut_points(build_block("moe")) == ["expert_out"]


def test_recurrence_and_pure_elementwise_do_not_cut():
    """A SEQUENTIAL recurrence (linear attention, Mamba scan) and a purely fusible chain (SwiGLU)
    carry no nonlinear/data-dependent barrier, so they stay a single fusion group."""
    for build in (build_linear_attention_block, build_mamba_block):
        assert determine_fusion_cut_points(build()) == []
    assert determine_fusion_cut_points(build_block("swiglu")) == []


def test_linear_contraction_is_not_a_barrier():
    """Only the nonlinear (softmax) reduction is a barrier; the linear PV contraction that also
    reduces the key axis is accumulator-streamable and must not appear as a cut."""
    cuts = barrier_cut_points(build_attention_block())
    assert cuts == ["softmax"]
    assert "context" not in cuts


def test_cnn_cut_points_carry_no_affine_barriers():
    """A conv graph has no nonlinear reduction or data-dependent read, so the barrier analysis adds
    nothing -- the CNN cut-point heuristic (MaxPool / Add+Relu) is unchanged."""
    assert barrier_cut_points(_conv_workload()) == []
