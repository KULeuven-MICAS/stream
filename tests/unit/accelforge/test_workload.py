"""Stream computation nodes as AccelForge Einsums: projections, priced inputs, names and what is refused."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineExpr, AffineMap

from stream.stages.estimation.accelforge.workload import UnsupportedConversion, describe, tensor_name
from stream.workload.node import ComputationNode, NormalizationNode
from stream.workload.tensor import Tensor

from .cores import conv


def _maps(*fns) -> tuple[AffineMap, ...]:
    return tuple(AffineMap.from_callable(fn) for fn in fns)


def _node(inputs, output, maps, kind: str = "Add", name: str = "node") -> ComputationNode:
    return ComputationNode(type=kind, name=name, inputs=tuple(inputs), outputs=(output,), operand_mapping=maps)


def _matmul(name: str = "mm") -> ComputationNode:
    a, b, c = Tensor.create("A", bf16, (32, 64)), Tensor.create("B", bf16, (64, 16)), Tensor.create("C", bf16, (32, 16))
    return _node(
        (a, b), c, _maps(lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n)), "MatMul", name
    )


def _projection(node: ComputationNode, sizes: tuple[int, ...], tensor: str) -> dict[str, str]:
    return dict(next(a.projection for a in describe(node, sizes).accesses if a.name == tensor))


def test_a_matmul_becomes_an_einsum_over_the_given_extents_with_its_ranks_bits_and_roles():
    einsum = describe(_matmul(), (8, 64, 16))
    assert einsum.sizes == (8, 64, 16)
    assert [dict(a.projection) for a in einsum.accesses] == [
        {"R0_A": "mm_d0", "R1_A": "mm_d1"},
        {"R0_B": "mm_d1", "R1_B": "mm_d2"},
        {"R0_C": "mm_d0", "R1_C": "mm_d2"},
    ]
    assert [(a.output, a.bits_per_value) for a in einsum.accesses] == [(False, 16), (False, 16), (True, 16)]
    assert dict(einsum.renames) == {"input": "A", "weight": "B", "output": "C", "partial_sums": "C"}


@pytest.mark.parametrize(
    "stride, window, sizes, projection",
    [
        (2, 3, (8, 4, 8, 8, 3, 3), {"R0_I": "conv_d1", "R1_I": "conv_d2 + conv_d4", "R2_I": "2*conv_d3 + conv_d5"}),
        (2, 1, (8, 4, 8, 8, 3, 1), {"R0_I": "conv_d1", "R1_I": "conv_d2 + conv_d4", "R2_I": "conv_d3"}),
    ],
    ids=["window-sums", "single-strided-dim-compacted"],
)
def test_a_conv_keeps_its_window_sums_and_a_single_strided_dim_becomes_dense(stride, window, sizes, projection):
    assert _projection(conv(stride, window), sizes, "I") == projection


def test_offsets_are_dropped_as_zigzag_drops_them_and_constant_indices_index_zero():
    h, o = Tensor.create("H", bf16, (5, 1)), Tensor.create("Hn", bf16, (4,))
    shifted = AffineMap(1, 0, (AffineExpr.dimension(0) + 1, AffineExpr.constant(0)))
    assert _projection(_node((h,), o, (shifted, *_maps(lambda t: (t,)))), (4,), "H") == {"R0_H": "node_d0", "R1_H": "0"}


def _gemm_with_bias() -> ComputationNode:
    mm = _matmul()
    bias = Tensor.create("bias", bf16, (16,))
    maps = (*mm.operand_mapping[:2], *_maps(lambda m, k, n: (n,)), mm.operand_mapping[2])
    return _node((*mm.inputs, bias), mm.outputs[0], maps, "Gemm")


def _layer_norm() -> ComputationNode:
    x, y = Tensor.create("x", bf16, (8, 16)), Tensor.create("y", bf16, (8, 16))
    scale, shift = Tensor.create("scale", bf16, (16,)), Tensor.create("shift", bf16, (16,))
    same, channel = _maps(lambda i, j: (i, j), lambda i, j: (j,))
    return NormalizationNode(
        type="LayerNormalization",
        name="norm",
        inputs=(x, scale, shift),
        outputs=(y,),
        operand_mapping=(same, channel, channel, same),
        reduction_axes=(1,),
    )


def _vector(*names: str) -> tuple[Tensor, ...]:
    return tuple(Tensor.create(name, bf16, (8,)) for name in names)


def _select() -> ComputationNode:
    a, b, c, o = _vector("a", "b", "c", "o")
    return _node((a, b, c), o, _maps(lambda i: (i,)) * 4, "Where")


def _square() -> ComputationNode:
    x, y = _vector("x", "y")
    return _node((x, x), y, _maps(lambda i: (i,)) * 3, "Mul")


@pytest.mark.parametrize(
    "node, sizes, accessed",
    [
        (_gemm_with_bias(), (32, 64, 16), ["A", "B", "C"]),
        (_layer_norm(), (8, 16), ["x", "y"]),
        (_select(), (8,), ["a", "b", "c", "o"]),
        (_square(), (8,), ["x", "y"]),
    ],
    ids=["mac-first-two-inputs", "normalization-data-alone", "other-node-every-input", "repeated-input-once"],
)
def test_the_inputs_priced_are_those_zigzag_prices(node, sizes, accessed):
    assert [a.name for a in describe(node, sizes).accesses] == accessed


def _read(expr: AffineExpr) -> ComputationNode:
    x, y = Tensor.create("x", bf16, (8,)), Tensor.create("y", bf16, (4, 4))
    return _node((x,), y, (AffineMap(2, 0, (expr,)), *_maps(lambda i, j: (i, j))), name="odd")


def _two_maps() -> ComputationNode:
    x, y = Tensor.create("x", bf16, (8, 8)), Tensor.create("y", bf16, (8, 8))
    return _node((x, x), y, _maps(lambda i, j: (i, j), lambda i, j: (j, i), lambda i, j: (i, j)), "Mul")


def _one_identifier() -> ComputationNode:
    first, second, out = _vector("a-b", "a_b", "o")
    return _node((first, second), out, _maps(lambda i: (i,)) * 3)


@pytest.mark.parametrize(
    "node, sizes, reason",
    [
        (_matmul(), (8, 64), "2 dim sizes given for 3 dims"),
        (conv(3, 2), (8, 4, 8, 8, 3, 2), "leaves gaps"),
        (_read(AffineExpr.dimension(0) % 2), (4, 4), "not affine-linear"),
        (_read(AffineExpr.dimension(0) // 2), (4, 4), "not affine-linear"),
        (
            _read(AffineBinaryOpExpr(AffineBinaryOpKind.Mul, AffineExpr.dimension(0), AffineExpr.dimension(1))),
            (4, 4),
            "multiplies two dims",
        ),
        (_two_maps(), (8, 8), "tensor x is accessed twice with different maps"),
        (_one_identifier(), (8,), "tensors a-b and a_b both become a_b"),
    ],
    ids=["extents", "stride-past-window", "mod", "floordiv", "product", "two-maps", "one-identifier"],
)
def test_a_node_no_einsum_can_express_is_refused_with_the_reason(node, sizes, reason):
    with pytest.raises(UnsupportedConversion, match=reason):
        describe(node, sizes)


def test_names_become_identifiers_clear_of_reserved_words():
    assert describe(_matmul(name="/layer.0/MatMul"), (32, 64, 16)).name == "n_layer_0_MatMul"
    assert tensor_name(Tensor.create("input", bf16, (1,))) == "t_input"
    assert tensor_name(Tensor.create("a-b", bf16, (1,))) == "a_b"
