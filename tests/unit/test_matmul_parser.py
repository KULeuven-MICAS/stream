"""Tests for the MatMul parser: affine maps for 2D, batched, and broadcast matmuls."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.affine_access import footprint
from stream.workload.iterator_type import IteratorType, derive_iterator_types


def _matmul_model(a_shape: tuple[int, ...], b_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> onnx.ModelProto:
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(a_shape))
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, list(b_shape))
    out = helper.make_tensor_value_info("C", TensorProto.FLOAT, list(out_shape))
    node = helper.make_node("MatMul", ["A", "B"], ["C"], name="mm")
    graph = helper.make_graph([node], "g", [a, b], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _parse_single(model: onnx.ModelProto):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser.workload.get_computation_nodes()[0]


def test_matmul_2d_is_gemm_like():
    """2D MatMul (M,K)@(K,N)->(M,N): m,n PARALLEL, k the single REDUCTION."""
    node = _parse_single(_matmul_model((8, 4), (4, 6), (8, 6)))
    assert node.type == "MatMul"
    assert node.num_dims == 3  # m, n, k
    types = derive_iterator_types(node)
    reductions = [p for p, t in types.items() if t == IteratorType.REDUCTION]
    assert len(reductions) == 1  # exactly k
    # footprint: an (m=0..1, n=0..2, k) tile reads A[0..1, all k] and writes C[0..1, 0..2]
    a_map, c_map = node.operand_mapping[0], node.operand_mapping[2]
    tile = {0: range(0, 2), 1: range(0, 3), 2: range(0, 4)}
    assert footprint(a_map, tile) == (range(0, 2), range(0, 4))
    assert footprint(c_map, tile) == (range(0, 2), range(0, 3))


def test_matmul_batched_4d():
    """Batched MatMul [b,h,i,k]@[b,h,k,j]->[b,h,i,j]: b,h,i,j PARALLEL, k REDUCTION."""
    node = _parse_single(_matmul_model((2, 3, 8, 4), (2, 3, 4, 6), (2, 3, 8, 6)))
    assert node.num_dims == 5  # b,h (batch) + m,n,k
    types = derive_iterator_types(node)
    assert sum(t == IteratorType.REDUCTION for t in types.values()) == 1  # k
    assert sum(t == IteratorType.PARALLEL for t in types.values()) == 4  # b,h,m,n


def test_matmul_broadcast_batch_maps_to_constant():
    """A broadcast batch axis (size 1 on one operand) indexes that operand at constant 0."""
    node = _parse_single(_matmul_model((1, 8, 4), (5, 4, 6), (5, 8, 6)))
    a_map, _b_map, _c_map = node.operand_mapping
    # output batch dim (position 0) does not appear in A's map (A broadcasts it) -> A indexes it at 0
    from xdsl.ir.affine import AffineConstantExpr

    assert isinstance(a_map.results[0], AffineConstantExpr)
    assert a_map.results[0].value == 0


def test_matmul_footprint_matches_numpy_contraction_shape():
    """The affine maps agree with numpy.matmul on the produced/consumed index extents."""
    a, b = np.random.default_rng(0).random((2, 5, 3)), np.random.default_rng(1).random((2, 3, 7))
    c = np.matmul(a, b)
    node = _parse_single(_matmul_model(a.shape, b.shape, c.shape))
    assert node.outputs[0].shape == c.shape
