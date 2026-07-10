"""End-to-end parse of the attention head ONNX fixture: affine MatMuls, a layout barrier, and a
fusible normalization.

The fixture (``attention_head.onnx``) is single-head attention exported from a 2-operand-MatMul
frontend: Q/K/V projections, a Transpose to form Kᵀ, the QKᵀ scores, a row-wise Softmax over the key
axis, and the context matmul. It exercises the parser additions: MatMul as an affine
ComputationNode, Transpose as a layout FusionEdge, and Softmax as a schedulable
NormalizationNode that fuses (rather than splitting the graph).
"""

from __future__ import annotations

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import FusionEdge, NormalizationNode

FIXTURE = "stream/inputs/testing/workload/attention_head.onnx"


def _workload():
    parser = ONNXModelParser(FIXTURE)
    parser.run()
    return parser.workload


def test_attention_parses_matmuls_layout_barrier_and_normalization():
    wl = _workload()
    matmuls = [n for n in wl.get_computation_nodes() if n.type == "MatMul"]
    barriers = [n for n in wl.nodes if isinstance(n, FusionEdge)]
    norms = [n for n in wl.get_computation_nodes() if isinstance(n, NormalizationNode)]
    assert len(matmuls) == 5  # Q, K, V projections + scores + context
    assert {b.op_type for b in barriers} == {"Transpose"}  # only the layout op is a barrier
    assert [n.type for n in norms] == ["Softmax"]  # the normalization is a compute node


def test_attention_scores_have_reduction_dim():
    """Every attention MatMul contracts exactly one dimension (the shared/contraction axis)."""
    wl = _workload()
    for mm in (n for n in wl.get_computation_nodes() if n.type == "MatMul"):
        types = derive_iterator_types(mm)
        assert sum(t == IteratorType.REDUCTION for t in types.values()) == 1


def test_normalization_fuses_into_the_score_region():
    """Softmax no longer splits the graph -- it sits inside a fusible region with the score
    and context matmuls (only the Transpose layout op remains a barrier). Every group resolves."""
    wl = _workload()
    groups = wl.split_fusion_groups()
    norm = next(n for n in wl.get_computation_nodes() if isinstance(n, NormalizationNode))
    region = next(g for g in groups if any(c.name == norm.name for c in g.get_computation_nodes()))
    region_types = {c.type for c in region.get_computation_nodes()}
    assert "MatMul" in region_types and "Softmax" in region_types  # fused together
    for g in groups:
        assert len(g.get_dimension_sizes()) > 0


def test_normalization_carries_its_reduction_axis():
    wl = _workload()
    norm = next(n for n in wl.get_computation_nodes() if isinstance(n, NormalizationNode))
    # the Softmax output is (81, 81); ONNX default axis=-1 -> reduces position 1
    assert norm.reduction_axes == (1,)
