"""The operator-decomposition registry (stream.workload.decompose) and the flash-attention block's
affine sub-operator decomposition.

One generic registry expands any decomposable op into its affine sub-operators, so the framework reasons
about the coarse kernel and its fine (array-vs-vector) dataflow uniformly -- a softmax, a flash-attention
block, and any future op the same way.
"""

from __future__ import annotations

import numpy as np
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.workload.blocks import build_block
from stream.workload.decompose import decompose, has_decomposition, register_decomposition
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.rewrites.flash_attention import decompose_attention_block
from stream.workload.rewrites.reference import direct_attention, online_softmax_attention
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload


def _attention_block():
    flash = build_block("flash_attention", seq_q=8, seq_k=32, d_head=8, block_size=16)
    return next(n for n in flash.get_computation_nodes() if n.type == "AttentionBlock")


def _sub(dec: Workload, suffix: str) -> ComputationNode:
    return next(n for n in dec.get_computation_nodes() if n.name.endswith(suffix))


def _reduction_extent(node: ComputationNode) -> int:
    """Size of the (single) axis this node reduces over -- read off the operand it fully contracts."""
    types = derive_iterator_types(node)
    (red_pos,) = [p for p, t in types.items() if t == IteratorType.REDUCTION]
    for tensor in node.tensors:
        for result, size in zip(node.get_mapping(tensor).results, tensor.shape, strict=True):
            if getattr(result, "position", None) == red_pos:
                return size
    raise AssertionError("no reduction extent found")


# --------------------------------------------------------------------------- #
#  The registry is generic + extensible                                       #
# --------------------------------------------------------------------------- #
def test_registry_decomposes_softmax_and_attention_block_uniformly():
    softmax = next(n for n in build_block("attention").get_computation_nodes() if n.type == "Softmax")
    assert has_decomposition(softmax)
    assert [n.type for n in decompose(softmax).get_computation_nodes()] == ["ReduceMax", "Exp", "ReduceSum", "Div"]

    assert has_decomposition(_attention_block())
    assert decompose(_attention_block()) is not None


def test_a_plain_matmul_has_no_decomposition():
    matmul = next(n for n in build_block("swiglu").get_computation_nodes() if n.type == "MatMul")
    assert not has_decomposition(matmul)
    assert decompose(matmul) is None


def test_a_new_op_is_one_registration():
    """The reuse seam: a future op registers a decomposer and everything downstream sees its sub-ops."""

    def _decompose_double(node: ComputationNode) -> Workload:
        x = node.inputs[0]
        y = Tensor.create(f"{node.name}_y", x.operand_type, x.shape)
        idn = AffineMap.identity(len(x.shape))
        sub = ComputationNode(
            type="Mul", name=f"{node.name}_mul", inputs=(x,), outputs=(y,), operand_mapping=(idn, idn)
        )
        return Workload([InEdge(name=x.name, outputs=(x,)), sub, OutEdge(name=f"{node.name}_out", inputs=(y,))])

    register_decomposition("MyFutureOp", _decompose_double)
    x = Tensor.create("x", bf16, (4, 8))
    node = ComputationNode(
        type="MyFutureOp", name="fut", inputs=(x,), outputs=(x,), operand_mapping=(AffineMap.identity(2),)
    )
    assert has_decomposition(node)
    assert [n.type for n in decompose(node).get_computation_nodes()] == ["Mul"]


# --------------------------------------------------------------------------- #
#  Flash-attention block: the two matmuls contract DIFFERENT axes             #
# --------------------------------------------------------------------------- #
def test_attention_block_exposes_two_matmuls_and_the_softmax_stats():
    dec = decompose_attention_block(_attention_block())
    ops = [n.type for n in dec.get_computation_nodes()]
    assert ops.count("MatMul") == 2  # scores (Q@Kᵀ) and context (P@V)
    assert {"ReduceMax", "Exp", "ReduceSum"} <= set(ops)  # the online-softmax stats


def test_scores_contracts_the_head_dim_and_context_the_key_dim():
    """The crux the opaque block cannot express: scores reduces the head dim d (size d_head), context
    reduces the key dim c (size block); d is PARALLEL in context (it is the output axis there)."""
    dec = decompose_attention_block(_attention_block())
    scores, context = _sub(dec, "_scores"), _sub(dec, "_context")
    assert _reduction_extent(scores) == 8  # d_head -- scores contracts the head dim
    assert _reduction_extent(context) == 16  # block -- context contracts the key dim
    # d (position 1 in context's (i, d, c) space) is PARALLEL there, REDUCTION in scores
    assert derive_iterator_types(context)[1] == IteratorType.PARALLEL
    assert IteratorType.REDUCTION in derive_iterator_types(scores).values()


def test_decomposition_math_matches_direct_attention():
    """The decomposition mirrors the online-softmax algorithm exactly, so the block chain is a faithful
    (not approximate) refinement of full attention."""
    rng = np.random.default_rng(0)
    q, k, v = rng.standard_normal((8, 4)), rng.standard_normal((32, 4)), rng.standard_normal((32, 4))
    np.testing.assert_allclose(online_softmax_attention(q, k, v, 16), direct_attention(q, k, v), atol=1e-9)
