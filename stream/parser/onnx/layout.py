"""Layout-only ONNX operators (Transpose, Reshape, Flatten, Squeeze, Unsqueeze), folded into the nodes that read
their output.

Such an operator moves no data an accelerator has to compute: a compiler lays the tensor out so that its reader
indexes the original one, a reshape being a different view of the same buffer and a transpose an operand layout or a
DMA's access pattern. Stream does the same with the affine access maps: the reader of a layout operator's output
reads its input instead, through the composed map. A transpose permutes the reader's indices, a reshape that splits an
axis indexes the original axis with an affine combination of the new ones, and one that merges axes splits the
reader's loop over the merged axis into one loop per original axis. Only where that is not possible (axes that are
split and merged at once, or a merged axis the reader does not walk with a single loop) is the operator materialized
as a ``FusionEdge``, a boundary its tensor crosses through memory.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

from stream.workload.node import ComputationNode, Node, NormalizationNode
from stream.workload.workload import Tensor

LAYOUT_OPS = frozenset({"Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze"})

AxisGroup = tuple[tuple[int, ...], tuple[int, ...]]
"""Axes of a layout operator's output and the axes of its input they view, the same elements in row-major order."""


@dataclass(frozen=True)
class Layout:
    """How the axes of a layout operator's ``output`` view those of its ``source``: per group of axes holding the same
    elements, which output axes and which source axes. Source axes of extent one in no group are indexed at 0; output
    axes of extent one in no group index nothing. ``groups`` is None when the axes cannot be grouped."""

    name: str
    op_type: str
    source: Tensor
    output: Tensor
    groups: tuple[AxisGroup, ...] | None


def transpose_groups(perm: list[int]) -> tuple[AxisGroup, ...]:
    return tuple(((axis,), (source,)) for axis, source in enumerate(perm))


def reshape_groups(source_shape: tuple[int, ...], shape: tuple[int, ...]) -> tuple[AxisGroup, ...] | None:
    """The groups of a row-major reshape: consecutive axes of either side whose extents multiply to the same number."""
    sources = [axis for axis, size in enumerate(source_shape) if size != 1]
    outputs = [axis for axis, size in enumerate(shape) if size != 1]
    groups: list[AxisGroup] = []
    i = j = 0
    while i < len(sources) and j < len(outputs):
        group_in, group_out = [sources[i]], [outputs[j]]
        size_in, size_out = source_shape[sources[i]], shape[outputs[j]]
        i, j = i + 1, j + 1
        while size_in != size_out:
            if size_in < size_out and i < len(sources):
                group_in.append(sources[i])
                size_in *= source_shape[sources[i]]
                i += 1
            elif size_out < size_in and j < len(outputs):
                group_out.append(outputs[j])
                size_out *= shape[outputs[j]]
                j += 1
            else:
                return None
        groups.append((tuple(group_out), tuple(group_in)))
    return tuple(groups) if i == len(sources) and j == len(outputs) else None


def _strides(sizes: list[int]) -> list[int]:
    return [math.prod(sizes[k + 1 :]) for k in range(len(sizes))]


def _split_dim(node: ComputationNode, dim: int, sizes: list[int]) -> tuple[ComputationNode, list[int]]:
    """``node`` with loop ``dim`` split into one loop per extent in ``sizes``, outermost first: the first keeps the
    position, the others are appended. Every operand indexing ``dim`` indexes their row-major combination instead."""
    num_dims = node.num_dims
    added = list(range(num_dims, num_dims + len(sizes) - 1))
    loops = [dim, *added]
    combined: AffineExpr = AffineExpr.constant(0)
    for loop, stride in zip(loops, _strides(sizes), strict=True):
        combined = combined + AffineExpr.dimension(loop) * stride
    substitution = AffineMap(
        num_dims + len(added),
        0,
        tuple(combined if d == dim else AffineExpr.dimension(d) for d in range(num_dims)),
    )
    maps = tuple(m.compose(substitution) for m in node.operand_mapping)
    return dataclasses.replace(node, operand_mapping=maps), loops


def fold(node: ComputationNode, index: int, layout: Layout) -> ComputationNode | None:
    """``node`` with its input ``index`` (``layout.output``) read from ``layout.source``, or None when the reader
    cannot index the source with an affine map."""
    if layout.groups is None:
        return None
    access = node.operand_mapping[index]
    merges: list[tuple[AxisGroup, int]] = []
    for group in layout.groups:
        outputs, sources = group
        if len(sources) > 1 and len(outputs) > 1:
            return None
        if len(sources) > 1:
            walked = access.results[outputs[0]]
            if not isinstance(walked, AffineDimExpr):
                return None
            merges.append((group, walked.position))
    dims = [dim for _, dim in merges]
    windows = {dim for dim, _ in node.window_extents}
    reduced = set(node.reduction_axes) if isinstance(node, NormalizationNode) else set()
    if len(set(dims)) != len(dims) or windows & set(dims) or reduced & set(dims):
        return None

    split_loops: dict[AxisGroup, list[int]] = {}
    for group, dim in merges:
        node, split_loops[group] = _split_dim(node, dim, [layout.source.shape[axis] for axis in group[1]])

    results = node.operand_mapping[index].results
    indices: list[AffineExpr] = [AffineExpr.constant(0)] * len(layout.source.shape)
    for group in layout.groups:
        outputs, sources = group
        if group in split_loops:
            for axis, loop in zip(sources, split_loops[group], strict=True):
                indices[axis] = AffineExpr.dimension(loop)
            continue
        combined: AffineExpr = AffineExpr.constant(0)
        for axis, stride in zip(outputs, _strides([layout.output.shape[a] for a in outputs]), strict=True):
            combined = combined + results[axis] * stride
        indices[sources[0]] = combined

    maps = list(node.operand_mapping)
    maps[index] = AffineMap(node.num_dims, 0, tuple(indices))
    inputs = list(node.inputs)
    inputs[index] = layout.source
    return dataclasses.replace(node, inputs=tuple(inputs), operand_mapping=tuple(maps))


def _linear_terms(expr: AffineExpr) -> dict[int, int] | None:
    """The coefficient of each loop in ``expr``, or None when it is not a sum of loops times constants."""
    if isinstance(expr, AffineDimExpr):
        return {expr.position: 1}
    if isinstance(expr, AffineConstantExpr):
        return {} if expr.value == 0 else None
    if isinstance(expr, AffineBinaryOpExpr) and expr.kind is AffineBinaryOpKind.Add:
        lhs, rhs = _linear_terms(expr.lhs), _linear_terms(expr.rhs)
        if lhs is None or rhs is None or lhs.keys() & rhs.keys():
            return None
        return lhs | rhs
    if isinstance(expr, AffineBinaryOpExpr) and expr.kind is AffineBinaryOpKind.Mul:
        for loop, scale in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            if isinstance(scale, AffineConstantExpr) and (terms := _linear_terms(loop)) is not None:
                return {d: c * scale.value for d, c in terms.items()}
    return None


def _loop_extents(node: ComputationNode) -> dict[int, int]:
    """The extent of each loop of ``node`` that indexes an operand axis on its own."""
    extents: dict[int, int] = {}
    for tensor, access in zip(node.tensors, node.operand_mapping, strict=True):
        for axis, result in enumerate(access.results):
            if isinstance(result, AffineDimExpr):
                extents.setdefault(result.position, tensor.shape[axis])
    return extents


def _mixed_radix(expr: AffineExpr, extents: dict[int, int], size: int) -> list[tuple[int, int]] | None:
    """The loops of ``expr``, outermost first, with their extents, when it walks an axis of ``size`` the way a reshape
    views it: each loop's coefficient the product of the extents of the loops inside it, together covering the axis
    once. The coefficients give the extents, which must agree with ``extents`` where it knows them."""
    terms = _linear_terms(expr)
    if terms is None or len(terms) < 2:  # noqa: PLR2004
        return None
    loops = sorted(terms, key=lambda d: -terms[d])
    outer = [size, *(terms[d] for d in loops[:-1])]
    if terms[loops[-1]] != 1 or any(o % terms[d] for o, d in zip(outer, loops, strict=True)):
        return None
    walked = [(d, o // terms[d]) for o, d in zip(outer, loops, strict=True)]
    return walked if all(extents.get(d, e) == e for d, e in walked) else None


def _refine_access(node: ComputationNode, name: str, axis: int, factors: list[int]) -> ComputationNode | None:
    """``node`` reading or writing axis ``axis`` of tensor ``name`` as ``len(factors)`` axes of those extents, its
    loop over the axis split to match where it walks it with one loop; None where it walks it otherwise."""
    while True:
        extents = _loop_extents(node)
        pending = [
            (p, access.results[axis])
            for p, (tensor, access) in enumerate(zip(node.tensors, node.operand_mapping, strict=True))
            if tensor.name == name
            and isinstance(access.results[axis], AffineDimExpr)
            and len(tensor.shape) > axis
            and extents.get(access.results[axis].position) == math.prod(factors)
        ]
        if not pending:
            break
        loop = pending[0][1].position
        if loop in {dim for dim, _ in node.window_extents}:
            return None
        node, loops = _split_dim(node, loop, factors)
        if isinstance(node, NormalizationNode) and loop in node.reduction_axes:
            node = dataclasses.replace(node, reduction_axes=(*node.reduction_axes, *loops[1:]))

    extents = _loop_extents(node)
    maps = list(node.operand_mapping)
    for p, (tensor, access) in enumerate(zip(node.tensors, node.operand_mapping, strict=True)):
        if tensor.name != name:
            continue
        result = access.results[axis]
        if isinstance(result, AffineConstantExpr) and result.value == 0:
            expanded: tuple[AffineExpr, ...] = (AffineExpr.constant(0),) * len(factors)
        elif (walked := _mixed_radix(result, extents, math.prod(factors))) and [e for _, e in walked] == factors:
            expanded = tuple(AffineExpr.dimension(d) for d, _ in walked)
        else:
            return None
        results = access.results
        maps[p] = AffineMap(node.num_dims, 0, (*results[:axis], *expanded, *results[axis + 1 :]))
    return dataclasses.replace(node, operand_mapping=tuple(maps))


def _with_tensor(node: Node, tensor: Tensor) -> Node:
    """``node`` with every tensor named as ``tensor`` replaced by it."""
    fields = [f for f in ("inputs", "outputs") if hasattr(node, f)]
    swap = {f: tuple(tensor if t.name == tensor.name else t for t in getattr(node, f)) for f in fields}
    return dataclasses.replace(node, **swap)


def _reshaped_axis(
    nodes: list[Node], touched: set[str], blocked: set
) -> tuple[Tensor, str, int, tuple[int, ...]] | None:
    """A tensor axis a ``touched`` node walks with a mixed-radix combination of loops, with their extents."""
    for node in nodes:
        if not isinstance(node, ComputationNode) or node.name not in touched:
            continue
        extents = _loop_extents(node)
        for tensor, access in zip(node.tensors, node.operand_mapping, strict=True):
            for axis, result in enumerate(access.results):
                walked = _mixed_radix(result, extents, tensor.shape[axis])
                if walked and (tensor.name, axis, factors := tuple(e for _, e in walked)) not in blocked:
                    return tensor, tensor.name, axis, factors
    return None


def factorize_axes(nodes: list[Node], touched: set[str]) -> list[Node]:
    """Normalize the access maps of ``nodes`` so that each operand axis is indexed by one loop: an axis a node walks
    with a mixed-radix combination of loops, as a folded reshape leaves it, becomes one axis per loop in every tensor
    and node that has it, a node walking it with a single loop splitting that loop to match. A reshape is then gone
    from the nodes as it is from the data, and every node reads like an einsum over its operands. An axis some node
    walks otherwise (a sliding window, say) keeps its combination. Only the nodes a fold ``touched``, and those
    refining their axes reaches, are normalized: a grouped convolution's channel index keeps the form its parser
    gives it."""
    blocked: set[tuple[str, int, tuple[int, ...]]] = set()
    touched = set(touched)
    while (found := _reshaped_axis(nodes, touched, blocked)) is not None:
        tensor, name, axis, factors = found
        refined = Tensor.create(name, tensor.operand_type, (*tensor.shape[:axis], *factors, *tensor.shape[axis + 1 :]))
        updated: list[Node] = []
        for node in nodes:
            rewritten = node
            if isinstance(node, ComputationNode) and any(t.name == name for t in node.tensors):
                rewritten = _refine_access(node, name, axis, list(factors))
                if rewritten is None:
                    blocked.add((name, axis, factors))
                    break
            has_tensor = any(t.name == name for t in _tensors(node))
            updated.append(_with_tensor(rewritten, refined) if has_tensor else rewritten)
        else:
            nodes = updated
            touched |= {n.name for n in nodes if any(t.name == name for t in _tensors(n))}
    return nodes


def _tensors(node: Node) -> tuple[Tensor, ...]:
    return (*getattr(node, "inputs", ()), *getattr(node, "outputs", ()))
