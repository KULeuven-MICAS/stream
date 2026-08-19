"""The affine fusion analysis drives where the generic pipeline splits fusion groups."""

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


def test_softmax_does_not_cut_attention_fuses_into_one_region():
    """MHA/GQA/KV-cache decode: the softmax is not a hard barrier, so attention stays one region."""
    for build in (build_attention_block, build_gqa_block, build_kv_cache_decode_step):
        assert determine_fusion_cut_points(build()) == []


def test_data_dependent_combine_is_a_cut_point_in_moe():
    """The MoE combine's data-dependent read is the only hard barrier and ends the expert GEMM group."""
    assert determine_fusion_cut_points(build_block("moe")) == ["expert_out"]
    assert barrier_cut_points(build_block("moe")) == ["expert_out"]


def test_recurrence_and_pure_elementwise_do_not_cut():
    """A SEQUENTIAL recurrence and a purely fusible chain (SwiGLU) carry no barrier: one fusion group."""
    for build in (build_linear_attention_block, build_mamba_block):
        assert determine_fusion_cut_points(build()) == []
    assert determine_fusion_cut_points(build_block("swiglu")) == []


def test_no_reduction_is_a_hard_barrier():
    """Neither the softmax reduction nor the PV contraction is a graph cut -- both fuse."""
    assert barrier_cut_points(build_attention_block()) == []


def test_cnn_cut_points_carry_no_affine_barriers():
    """A conv graph has no data-dependent read, so the barrier analysis adds no cut points."""
    assert barrier_cut_points(_conv_workload()) == []
