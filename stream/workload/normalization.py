"""Normalization ops (Softmax, LpNormalization, …) decomposed into affine sub-ops that reduce over
``reduction_axes`` and are element-wise (freely fusible) over the parallel axes."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap

from stream.workload.node import ComputationNode, InEdge, NormalizationNode, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = [
    "reduction_axes",
    "parallel_axes",
    "decompose_normalization",
    "expand_normalizations",
    "collapse_fused_kernels",
    "fused_kernel_tag",
    "softmax_reference",
    "NORMALIZATION_OPS",
    "REDUCTION_SUBOPS",
]

# Sub-operator types that carry the normalization's intra-op reduction (they drop the reduced axes).
REDUCTION_SUBOPS = ("ReduceMax", "ReduceSum", "ReduceSumSquare")

# Separator in a ``fused_kernel`` tag: ``"<OpType>:<name>"`` (e.g. ``"Softmax:softmax"``).
FUSED_KERNEL_SEP = ":"


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


def _softmax_subgraph(
    x: Tensor,
    axes: tuple[int, ...],
    dt: FixedBitwidthType,
    base: str,
    out: Tensor | None = None,
    tag: str | None = None,
) -> tuple[list, Tensor]:
    """Safe softmax: max → exp(·−max) → sum → div. Two reductions over ``axes``, two broadcasts.

    ``out`` (when given) is the tensor the final Div writes -- so an in-graph expansion reuses the
    normalization's original output and downstream consumers are unchanged; ``tag`` labels every sub-op
    with its origin fused kernel."""
    rank = len(x.shape)
    idn, drp = _identity(rank), _drop(rank, axes)
    red = _reduced_shape(x.shape, axes)
    m = Tensor.create(f"{base}_max", dt, red)
    e = Tensor.create(f"{base}_exp", dt, x.shape)
    s = Tensor.create(f"{base}_sum", dt, red)
    y = out if out is not None else Tensor.create(f"{base}_out", dt, x.shape)
    kw = {"fused_kernel": tag}
    nodes = [
        ComputationNode(
            type="ReduceMax", name=f"{base}_max", inputs=(x,), outputs=(m,), operand_mapping=(idn, drp), **kw
        ),
        ComputationNode(
            type="Exp", name=f"{base}_exp", inputs=(x, m), outputs=(e,), operand_mapping=(idn, drp, idn), **kw
        ),
        ComputationNode(
            type="ReduceSum", name=f"{base}_sum", inputs=(e,), outputs=(s,), operand_mapping=(idn, drp), **kw
        ),
        ComputationNode(
            type="Div", name=f"{base}_div", inputs=(e, s), outputs=(y,), operand_mapping=(idn, drp, idn), **kw
        ),
    ]
    return nodes, y


def _lpnorm_subgraph(
    x: Tensor,
    axes: tuple[int, ...],
    dt: FixedBitwidthType,
    base: str,
    out: Tensor | None = None,
    tag: str | None = None,
) -> tuple[list, Tensor]:
    """L2 normalization: sum(x²) over ``axes`` → sqrt → div. One reduction, one broadcast."""
    rank = len(x.shape)
    idn, drp = _identity(rank), _drop(rank, axes)
    red = _reduced_shape(x.shape, axes)
    red_rank = len(red)
    s = Tensor.create(f"{base}_sumsq", dt, red)
    norm = Tensor.create(f"{base}_norm", dt, red)
    y = out if out is not None else Tensor.create(f"{base}_out", dt, x.shape)
    kw = {"fused_kernel": tag}
    nodes = [
        ComputationNode(
            type="ReduceSumSquare", name=f"{base}_sumsq", inputs=(x,), outputs=(s,), operand_mapping=(idn, drp), **kw
        ),
        ComputationNode(
            type="Sqrt",
            name=f"{base}_sqrt",
            inputs=(s,),
            outputs=(norm,),
            operand_mapping=(_identity(red_rank), _identity(red_rank)),
            **kw,
        ),
        ComputationNode(
            type="Div", name=f"{base}_div", inputs=(x, norm), outputs=(y,), operand_mapping=(idn, drp, idn), **kw
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


def fused_kernel_tag(node: NormalizationNode) -> str:
    """The ``fused_kernel`` label carried by every sub-op of ``node`` once expanded (``"<type>:<name>"``)."""
    return f"{node.type}{FUSED_KERNEL_SEP}{node.name}"


def expand_normalizations(workload: Workload) -> Workload:
    """Replace every ``NormalizationNode`` in ``workload`` with its affine sub-operator subgraph, in place.

    The sub-ops consume the normalization's input and the final op writes its *original* output tensor, so
    every downstream consumer is unchanged; each sub-op is tagged with :func:`fused_kernel_tag` so the
    group is re-collapsible for codegen. This is the cost/fusion view: the two reduction passes of a safe
    softmax (max, sum) become explicit affine reductions the cost model counts and the fusion analysis
    reads directly, instead of an identity-mapped node that under-counts to one element-wise pass.
    Non-normalization nodes pass through unchanged."""
    new_nodes: list = []
    for node in nx.lexicographical_topological_sort(workload, key=lambda n: n.name):
        if isinstance(node, NormalizationNode):
            builder = _SUBGRAPHS.get(node.type)
            if builder is None:
                raise NotImplementedError(f"no affine decomposition registered for normalization {node.type!r}")
            x = node.inputs[0]
            subnodes, _ = builder(
                x, node.reduction_axes, x.operand_type, node.name, out=node.outputs[0], tag=fused_kernel_tag(node)
            )
            new_nodes.extend(subnodes)
        else:
            new_nodes.append(node)
    return Workload(new_nodes)


def collapse_fused_kernels(workload: Workload) -> Workload:
    """Inverse of :func:`expand_normalizations`: regroup the sub-ops that share a ``fused_kernel`` tag back
    into their single ``NormalizationNode``.

    Proves the decomposition is reversible -- the tagged sub-op representation is sufficient to reconstruct
    the native kernel a codegen backend emits (the reduction axes are recovered from the reduce sub-op's
    dropped axes; the type and name from the tag). Untagged nodes pass through unchanged."""
    groups: dict[str, list[ComputationNode]] = defaultdict(list)
    passthrough: list = []
    for node in nx.lexicographical_topological_sort(workload, key=lambda n: n.name):
        tag = getattr(node, "fused_kernel", None)
        if tag and isinstance(node, ComputationNode):
            groups[tag].append(node)
        else:
            passthrough.append(node)

    new_nodes: list = list(passthrough)
    for tag, subs in groups.items():
        norm_type, _, norm_name = tag.partition(FUSED_KERNEL_SEP)
        produced = {t.name for s in subs for t in s.outputs}
        consumed = {t.name for s in subs for t in s.inputs}
        x = next(t for s in subs for t in s.inputs if t.name not in produced)
        y = next(t for s in subs for t in s.outputs if t.name not in consumed)
        reduce_op = next(s for s in subs if s.type in REDUCTION_SUBOPS)
        rank = reduce_op.num_dims
        kept = {r.position for r in reduce_op.get_mapping(reduce_op.outputs[0]).results if isinstance(r, AffineDimExpr)}
        reduction_axes = tuple(sorted(set(range(rank)) - kept))
        idn = _identity(rank)
        new_nodes.append(
            NormalizationNode(
                type=norm_type,
                name=norm_name,
                inputs=(x,),
                outputs=(y,),
                operand_mapping=(idn, idn),
                reduction_axes=reduction_axes,
            )
        )
    return Workload(new_nodes)


def softmax_reference(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy safe-softmax golden that :func:`_softmax_subgraph` mirrors."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
