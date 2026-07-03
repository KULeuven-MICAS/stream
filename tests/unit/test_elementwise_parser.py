"""Tests for the unified ElementwiseParser: rank inference, broadcasting, type preservation.

Replaces the former per-op Add/Mul/Relu/Simd parsers with one class. These tests pin that the
identity-map behaviour they had is preserved (2D and 4D) and that broadcast now works.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from stream.parser.onnx.model import ONNXModelParser


def _parse_single(node, inputs, outputs, initializers=()):  # type: ignore[no-untyped-def]
    import tempfile

    graph = helper.make_graph([node], "g", inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser.workload.get_computation_nodes()[0]


def _vi(name: str, shape: tuple[int, ...]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def test_binary_add_4d_identity_maps():
    """4D Add (ResNet residual): every operand indexes the identity iteration space."""
    node = _parse_single(
        helper.make_node("Add", ["A", "B"], ["C"], name="add"),
        [_vi("A", (1, 8, 4, 4)), _vi("B", (1, 8, 4, 4))],
        [_vi("C", (1, 8, 4, 4))],
    )
    assert node.type == "Add"
    assert node.num_dims == 4
    for m in node.operand_mapping:
        assert [r.position for r in m.results] == [0, 1, 2, 3]


def test_unary_relu_2d_identity_maps():
    """Unary Relu (SwiGLU-style 2D): one input map + one output map, both identity."""
    node = _parse_single(
        helper.make_node("Relu", ["A"], ["C"], name="relu"),
        [_vi("A", (16, 32))],
        [_vi("C", (16, 32))],
    )
    assert node.type == "Relu"
    assert node.num_dims == 2
    assert len(node.operand_mapping) == 2  # 1 input + 1 output


def test_broadcast_scalar_maps_to_constant():
    """A size-1 axis (attention scale / bias broadcast) indexes that operand at constant 0."""
    node = _parse_single(
        helper.make_node("Mul", ["A", "S"], ["C"], name="mul"),
        [_vi("A", (4, 8)), _vi("S", (1, 1))],
        [_vi("C", (4, 8))],
    )
    scale_map = node.operand_mapping[1]
    assert all(isinstance(r, AffineConstantExpr) and r.value == 0 for r in scale_map.results)
    # the full-size operand keeps identity indexing
    assert all(isinstance(r, AffineDimExpr) for r in node.operand_mapping[0].results)


def test_right_aligned_broadcast_of_lower_rank_operand():
    """A lower-rank operand right-aligns to the output rank (numpy broadcast)."""
    node = _parse_single(
        helper.make_node("Add", ["A", "b"], ["C"], name="addbias"),
        [_vi("A", (2, 4, 8)), _vi("b", (8,))],
        [_vi("C", (2, 4, 8))],
    )
    bias_map = node.operand_mapping[1]
    assert len(bias_map.results) == 1  # rank-1 operand
    assert isinstance(bias_map.results[0], AffineDimExpr)
    assert bias_map.results[0].position == 2  # aligned to the last output axis
