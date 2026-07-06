"""Tests for AccessRelation-aware fusion wiring (plan/06 + plan/11).

Data-dependent reads (MoE dispatch/combine) are hard fusion barriers; a normalization's reduced axis
is a per-axis barrier that still fuses along the parallel axes. The FusionAnalysisStage annotates the
context without changing allocation behaviour.
"""

from __future__ import annotations

from functools import cache

from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.generation.fusion_analysis import FusionAnalysisStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.blocks import build_block
from stream.workload.fusion.analysis import edge_fusions, workload_fusion_edges
from stream.workload.models import build_attention_block

CONV_FIXTURE = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"


@cache
def _conv_workload():
    parser = ONNXModelParser(CONV_FIXTURE)
    parser.run()
    return parser.workload


def _edge(edges, consumer, tensor=None):
    return next(e for e in edges if e.consumer == consumer and (tensor is None or e.tensor == tensor))


def test_attention_softmax_edge_is_a_reduction_axis_not_a_hard_barrier():
    edges = workload_fusion_edges(build_attention_block())
    softmax_edge = _edge(edges, "softmax")  # scores -> softmax
    assert softmax_edge.reduction_axes == (3,)  # reduces the key axis
    assert softmax_edge.data_dependent is False
    assert softmax_edge.fusible  # still fusible along the parallel axes (b, h, i)
    # no attention edge is a data-dependent barrier
    assert all(not e.data_dependent for e in edges)


def test_moe_expert_to_combine_is_a_hard_data_dependent_barrier():
    edges = workload_fusion_edges(build_block("moe", tokens=8, experts=2, capacity=4))
    combine_edge = _edge(edges, "combine", tensor="expert_out")  # expert_out -> combine (scatter)
    assert combine_edge.data_dependent is True
    assert combine_edge.fusible is False
    # the routing logits feeding combine are read affinely (only the data input is data-dependent)
    assert _edge(edges, "combine", tensor="router_logits").data_dependent is False
    # the dispatched -> expert GEMM edge is an ordinary affine fusion
    expert_edge = _edge(edges, "expert_in")
    assert expert_edge.data_dependent is False and expert_edge.fusible


def test_conv_chain_edges_are_plain_affine_fusions():
    edges = workload_fusion_edges(_conv_workload())
    assert edges  # there is at least one conv->conv edge
    assert all(not e.data_dependent and e.reduction_axes == () for e in edges)


def test_edge_fusions_handles_a_single_edge():
    wl = build_block("swiglu")
    gate = next(n for n in wl.get_computation_nodes() if n.name == "gate_proj")
    mul = next(n for n in wl.get_computation_nodes() if n.name == "gate_mul")
    silu = next(n for n in wl.get_computation_nodes() if n.name == "silu")
    # gate_proj -> silu shares exactly the gate tensor
    verdicts = edge_fusions(gate, silu)
    assert len(verdicts) == 1 and verdicts[0].fusible
    assert edge_fusions(gate, mul) == []  # gate_proj does not feed gate_mul directly


def test_fusion_analysis_stage_annotates_context():
    wl = build_attention_block()
    ctx = StageContext.from_kwargs(workload=wl)
    result = MainStage([FusionAnalysisStage, LeafStage], ctx).run()
    assert len(result) == 1
    edges = result[0].get("fusion_edges")
    assert edges and any(e.consumer == "softmax" for e in edges)
