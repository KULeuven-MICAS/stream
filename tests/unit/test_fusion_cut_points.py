"""The affine fusion analysis drives where the generic pipeline splits fusion groups.

Only a HARD barrier cuts the graph: a data-dependent read (MoE dispatch/combine, gather), whose
footprint depends on runtime data. A reduction -- linear (a MatMul contraction) OR nonlinear
(Softmax/LayerNorm) -- does NOT cut: the chain fuses, and the reduction is merely an illegal
tiling/streaming axis (kept resident), enforced by the tiling guards, not by a cut. So an attention
head fuses into one region instead of spilling its softmax output to memory. CNN graphs keep the
MaxPool / Add+Relu heuristic cuts.
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


def test_softmax_does_not_cut_attention_fuses_into_one_region():
    """MHA/GQA/KV-cache decode: the softmax is a nonlinear reduction, not a hard barrier, so the whole
    attention chain stays one fusion region (the reduction axis is kept resident, not spilled)."""
    for build in (build_attention_block, build_gqa_block, build_kv_cache_decode_step):
        assert determine_fusion_cut_points(build()) == []


def test_data_dependent_combine_is_a_cut_point_in_moe():
    """The MoE combine reads its input via a runtime routing table -- the only hard barrier -- so the
    producing expert GEMM ends its group."""
    assert determine_fusion_cut_points(build_block("moe")) == ["expert_out"]
    assert barrier_cut_points(build_block("moe")) == ["expert_out"]


def test_recurrence_and_pure_elementwise_do_not_cut():
    """A SEQUENTIAL recurrence (linear attention, Mamba scan) and a purely fusible chain (SwiGLU)
    carry no hard barrier, so they stay a single fusion group."""
    for build in (build_linear_attention_block, build_mamba_block):
        assert determine_fusion_cut_points(build()) == []
    assert determine_fusion_cut_points(build_block("swiglu")) == []


def test_no_reduction_is_a_hard_barrier():
    """Neither the nonlinear softmax reduction nor the linear PV contraction is a graph cut -- both
    fuse; only their reduced axis is off-limits for tiling (handled by the tiling guards)."""
    assert barrier_cut_points(build_attention_block()) == []


def test_cnn_cut_points_carry_no_affine_barriers():
    """A conv graph has no data-dependent read, so the barrier analysis adds nothing -- the CNN
    cut-point heuristic (MaxPool / Add+Relu) is unchanged."""
    assert barrier_cut_points(_conv_workload()) == []
