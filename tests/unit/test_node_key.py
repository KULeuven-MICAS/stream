"""Tests for the canonical node key: soundness (sensitivity) and dimension-rename invariance."""

from __future__ import annotations

from xdsl.dialects.builtin import bf16, i8
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.workload.node import ComputationNode
from stream.workload.node_key import node_key
from stream.workload.tensor import Tensor


def _gemm(name: str, m: int, k: int, n: int, dtype=bf16, op_type: str = "Gemm") -> ComputationNode:
    a = Tensor.create("A", dtype, (m, k))
    b = Tensor.create("B", dtype, (k, n))
    o = Tensor.create("O", dtype, (m, n))
    maps = (
        AffineMap.from_callable(lambda m, k, n: (m, k)),
        AffineMap.from_callable(lambda m, k, n: (k, n)),
        AffineMap.from_callable(lambda m, k, n: (m, n)),
    )
    return ComputationNode(type=op_type, name=name, inputs=(a, b), outputs=(o,), operand_mapping=maps)


def test_identical_nodes_share_key():
    assert node_key(_gemm("g1", 8, 4, 16)) == node_key(_gemm("g2", 8, 4, 16))


def test_op_type_is_part_of_identity():
    """Regression for AUDIT §4: a Conv and a Gemm with identical tensor shapes must not collide."""
    same_shapes_gemm = _gemm("g", 8, 4, 16, op_type="Gemm")
    same_shapes_conv = _gemm("c", 8, 4, 16, op_type="Conv")
    assert node_key(same_shapes_gemm) != node_key(same_shapes_conv)
    assert not same_shapes_gemm.has_same_performance(same_shapes_conv)


def test_precision_and_bounds_are_sensitive():
    base = _gemm("g", 8, 4, 16)
    assert node_key(base) != node_key(_gemm("g", 8, 4, 16, dtype=i8))  # precision
    assert node_key(base) != node_key(_gemm("g", 8, 8, 16))  # loop bound


def test_affine_map_is_sensitive():
    base = _gemm("g", 8, 4, 16)
    a = Tensor.create("A", bf16, (8, 4))
    b = Tensor.create("B", bf16, (4, 16))
    o = Tensor.create("O", bf16, (8, 16))
    transposed_b = (
        AffineMap.from_callable(lambda m, k, n: (m, k)),
        AffineMap.from_callable(lambda m, k, n: (n, k)),  # different map, same shapes
        AffineMap.from_callable(lambda m, k, n: (m, n)),
    )
    other = ComputationNode(type="Gemm", name="g", inputs=(a, b), outputs=(o,), operand_mapping=transposed_b)
    assert node_key(base) != node_key(other)


def test_dimension_rename_invariance():
    """A consistent permutation of the iteration dimensions yields the same key."""
    a = Tensor.create("A", bf16, (8, 4))
    b = Tensor.create("B", bf16, (4, 16))
    o = Tensor.create("O", bf16, (8, 16))

    def maps(a_pos, b_pos, o_pos):
        return tuple(AffineMap(3, 0, tuple(AffineExpr.dimension(p) for p in pos)) for pos in (a_pos, b_pos, o_pos))

    standard = ComputationNode(
        type="Gemm", name="g", inputs=(a, b), outputs=(o,), operand_mapping=maps([0, 1], [1, 2], [0, 2])
    )
    # relabel m->2, k->0, n->1 consistently across all operands
    renamed = ComputationNode(
        type="Gemm", name="g", inputs=(a, b), outputs=(o,), operand_mapping=maps([2, 0], [0, 1], [2, 1])
    )
    assert node_key(standard) == node_key(renamed)
