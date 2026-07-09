"""Normalization ops (Softmax, LpNormalization, …) decomposed into affine sub-ops that reduce over
``reduction_axes`` and are element-wise (freely fusible) over the parallel axes."""

from __future__ import annotations

import numpy as np
from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.workload.node import ComputationNode, InEdge, NormalizationNode, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = [
    "reduction_axes",
    "parallel_axes",
    "decompose_normalization",
    "softmax_reference",
    "NORMALIZATION_OPS",
]


def reduction_axes(node: ComputationNode) -> tuple[int, ...]:
    """The iteration-space positions the normalization reduces over (empty for a non-normalization)."""
    return node.reduction_axes if isinstance(node, NormalizationNode) else ()


def parallel_axes(node: ComputationNode) -> tuple[int, ...]:
    """The freely-fusible (element-wise) axes of a normalization: every axis but the reduction ones."""
    reduced = set(reduction_axes(node))
    return tuple(p for p in range(node.num_dims) if p not in reduced)


def _identity(rank: int) -> AffineMap:
    return AffineMap.identity(rank)


def _drop(rank: int, axes: tuple[int, ...]) -> AffineMap:
    """Access map of a reduced statistic: the ``rank``-dim iteration space with ``axes`` dropped."""
    kept = [i for i in range(rank) if i not in axes]
    return AffineMap(rank, 0, tuple(AffineExpr.dimension(i) for i in kept))


def _reduced_shape(shape: tuple[int, ...], axes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(s for i, s in enumerate(shape) if i not in axes)


def _softmax_subgraph(x: Tensor, axes: tuple[int, ...], dt: FixedBitwidthType, base: str) -> tuple[list, Tensor]:
    """Safe softmax: max → exp(·−max) → sum → div. Two reductions over ``axes``, two broadcasts."""
    rank = len(x.shape)
    idn, drp = _identity(rank), _drop(rank, axes)
    red = _reduced_shape(x.shape, axes)
    m = Tensor.create(f"{base}_max", dt, red)
    e = Tensor.create(f"{base}_exp", dt, x.shape)
    s = Tensor.create(f"{base}_sum", dt, red)
    y = Tensor.create(f"{base}_out", dt, x.shape)
    nodes = [
        ComputationNode(type="ReduceMax", name=f"{base}_max", inputs=(x,), outputs=(m,), operand_mapping=(idn, drp)),
        ComputationNode(type="Exp", name=f"{base}_exp", inputs=(x, m), outputs=(e,), operand_mapping=(idn, drp, idn)),
        ComputationNode(type="ReduceSum", name=f"{base}_sum", inputs=(e,), outputs=(s,), operand_mapping=(idn, drp)),
        ComputationNode(type="Div", name=f"{base}_div", inputs=(e, s), outputs=(y,), operand_mapping=(idn, drp, idn)),
    ]
    return nodes, y


def _lpnorm_subgraph(x: Tensor, axes: tuple[int, ...], dt: FixedBitwidthType, base: str) -> tuple[list, Tensor]:
    """L2 normalization: sum(x²) over ``axes`` → sqrt → div. One reduction, one broadcast."""
    rank = len(x.shape)
    idn, drp = _identity(rank), _drop(rank, axes)
    red = _reduced_shape(x.shape, axes)
    red_rank = len(red)
    s = Tensor.create(f"{base}_sumsq", dt, red)
    norm = Tensor.create(f"{base}_norm", dt, red)
    y = Tensor.create(f"{base}_out", dt, x.shape)
    nodes = [
        ComputationNode(
            type="ReduceSumSquare", name=f"{base}_sumsq", inputs=(x,), outputs=(s,), operand_mapping=(idn, drp)
        ),
        ComputationNode(
            type="Sqrt",
            name=f"{base}_sqrt",
            inputs=(s,),
            outputs=(norm,),
            operand_mapping=(_identity(red_rank), _identity(red_rank)),
        ),
        ComputationNode(
            type="Div", name=f"{base}_div", inputs=(x, norm), outputs=(y,), operand_mapping=(idn, drp, idn)
        ),
    ]
    return nodes, y


_SUBGRAPHS = {
    "Softmax": _softmax_subgraph,
    "LpNormalization": _lpnorm_subgraph,
}

NORMALIZATION_OPS = tuple(_SUBGRAPHS)


def decompose_normalization(node: NormalizationNode) -> Workload:
    """Expand a normalization into its affine sub-operator subgraph; raises ``NotImplementedError``
    if the op's sub-op math is not registered."""
    builder = _SUBGRAPHS.get(node.type)
    if builder is None:
        raise NotImplementedError(f"no affine decomposition registered for normalization {node.type!r}")
    x = node.inputs[0]
    subnodes, y = builder(x, node.reduction_axes, x.operand_type, node.name)
    return Workload([InEdge(name=x.name, outputs=(x,)), *subnodes, OutEdge(name=f"{node.name}_out", inputs=(y,))])


def softmax_reference(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy safe-softmax golden that :func:`_softmax_subgraph` mirrors."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
