"""Tests for the normalization representation: a single schedulable node + its affine decomposition.

Softmax is one NormalizationNode for scheduling (a native kernel), but decomposes under the hood into
max→exp→sum→div for fusion analysis, where the reduction axis is explicit and every other axis is
parallel (fusible).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import NormalizationNode
from stream.workload.normalization import (
    decompose_normalization,
    expanded_work,
    parallel_axes,
    reduction_axes,
    softmax_reference,
)
from stream.workload.rewrites.flash_attention import build_attention_chain
from stream.workload.tensor import Tensor


def _softmax_node(shape: tuple[int, ...], axis: int) -> NormalizationNode:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(shape))
    node = helper.make_node("Softmax", ["X"], ["Y"], name="sm", axis=axis)
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]), opset_imports=[helper.make_opsetid("", 17)])
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser.workload.get_computation_nodes()[0]


def test_softmax_parses_to_normalization_node_with_axis():
    node = _softmax_node((2, 8, 16, 16), axis=-1)
    assert isinstance(node, NormalizationNode)
    assert node.type == "Softmax"
    assert reduction_axes(node) == (3,)  # -1 resolves to the last of 4 axes
    assert parallel_axes(node) == (0, 1, 2)


def test_softmax_positive_axis():
    node = _softmax_node((4, 10), axis=1)
    assert reduction_axes(node) == (1,)


def test_scheduling_view_is_a_single_identity_node():
    """The node itself is one op with identity maps (the fused-kernel view): every dim indexes the
    output, so on the node alone there is no REDUCTION -- that lives in the decomposition."""
    node = _softmax_node((2, 8, 16, 16), axis=-1)
    assert len(node.operand_mapping) == 2  # one input + one output
    assert all(t == IteratorType.PARALLEL for t in derive_iterator_types(node).values())


def test_decomposition_exposes_the_reduction():
    """max→exp→sum→div; the two reduce sub-ops contract exactly the softmax axis, the two pointwise
    sub-ops are fully parallel."""
    node = _softmax_node((2, 8, 16, 16), axis=-1)
    dec = decompose_normalization(node)
    by_type = {n.type: n for n in dec.get_computation_nodes()}
    assert set(by_type) == {"ReduceMax", "Exp", "ReduceSum", "Div"}
    for reduce_op in ("ReduceMax", "ReduceSum"):
        types = derive_iterator_types(by_type[reduce_op])
        assert [p for p, t in types.items() if t == IteratorType.REDUCTION] == [3]
    for pointwise in ("Exp", "Div"):
        types = derive_iterator_types(by_type[pointwise])
        assert all(t == IteratorType.PARALLEL for t in types.values())
    # the decomposition is a self-contained workload whose dimension sizes resolve
    assert len(dec.get_dimension_sizes()) > 0


def test_decomposition_stat_tensors_drop_the_reduction_axis():
    node = _softmax_node((2, 8, 16, 16), axis=-1)
    dec = decompose_normalization(node)
    reduce_max = next(n for n in dec.get_computation_nodes() if n.type == "ReduceMax")
    assert reduce_max.outputs[0].shape == (2, 8, 16)  # the key axis is contracted away


def test_softmax_reference_normalizes():
    x = np.random.default_rng(1).random((3, 7))
    y = softmax_reference(x, axis=-1)
    np.testing.assert_allclose(y.sum(axis=-1), np.ones(3))
    assert np.all(y >= 0)


def test_lpnorm_decomposition_registered():
    """A NormalizationNode built directly (L2) decomposes into sum-of-squares → sqrt → div."""
    identity = AffineMap.identity(2)
    node = NormalizationNode(
        type="LpNormalization",
        name="l2",
        inputs=(Tensor.create("x", bf16, (4, 16)),),
        outputs=(Tensor.create("y", bf16, (4, 16)),),
        operand_mapping=(identity, identity),
        reduction_axes=(1,),
    )
    dec = decompose_normalization(node)
    assert {n.type for n in dec.get_computation_nodes()} == {"ReduceSumSquare", "Sqrt", "Div"}


def test_layernorm_is_expanded_and_priced_as_its_parts():
    """Mean, centre, squares, divide, scale and shift on every element of a row of 16; the square root once per row."""
    identity = AffineMap.identity(2)
    row = AffineMap(2, 0, (AffineExpr.dimension(1),))
    node = NormalizationNode(
        type="LayerNormalization",
        name="ln",
        inputs=(
            Tensor.create("x", bf16, (4, 16)),
            Tensor.create("scale", bf16, (16,)),
            Tensor.create("b", bf16, (16,)),
        ),
        outputs=(Tensor.create("y", bf16, (4, 16)),),
        operand_mapping=(identity, row, row, identity),
        reduction_axes=(1,),
    )
    assert expanded_work(node) == [
        ("ReduceMean", 1.0),
        ("Sub", 1.0),
        ("ReduceSumSquare", 1.0),
        ("Sqrt", 1 / 16),
        ("Div", 1.0),
        ("Mul", 1.0),
        ("Add", 1.0),
    ]
    scale = next(n for n in decompose_normalization(node).get_computation_nodes() if n.type == "Mul")
    assert [t.name for t in scale.inputs] == ["ln_norm", "scale"]


def test_an_attention_block_is_priced_as_its_two_matmuls_and_its_online_softmax():
    """Matmuls on every point of (query, key, head 64), the softmax per score, the statistics per query row."""
    block = build_attention_chain("a", 256, 256, 64, 64, bf16).get_computation_nodes()[0]
    parts = expanded_work(block)
    assert {op: sum(ops for name, ops in parts if name == op) for op, _ in parts} == {
        "MatMul": 2.0,
        "ReduceMax": 1 / 64,
        "Maximum": 1 / 64**2,
        "Exp": 1 / 64 + 1 / 64**2,
        "ReduceSum": 1 / 64,
        "ScaleAdd": 1 / 64 + 1 / 64**2,
    }
