"""Algorithmic iterator types derived from a node's affine operand maps.

PARALLEL indexes the output (freely tileable and spatially unrollable); REDUCTION is an
accumulation dim (indexes an input but not the output); SEQUENTIAL carries a cross-iteration
state dependence (read at ``t-k``, written at ``t``) and must not be spatially unrolled.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, auto

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

from stream.workload.affine_access import map_dim_positions
from stream.workload.node import HasIterationSpace
from stream.workload.tensor import Tensor

__all__ = [
    "IteratorType",
    "SequentialUnrollError",
    "NonlinearReductionUnrollError",
    "ReductionUnrollError",
    "derive_iterator_types",
    "sequential_dims",
    "nonlinear_reduction_dims",
    "is_state_operand",
    "streamed_operands",
    "check_spatial_unroll_legal",
    "check_spatial_unroll_accumulation_free",
]


class IteratorType(Enum):
    PARALLEL = auto()
    REDUCTION = auto()
    SEQUENTIAL = auto()


class SequentialUnrollError(ValueError):
    """Raised when a SEQUENTIAL iteration dimension is assigned to spatial unrolling."""


class NonlinearReductionUnrollError(ValueError):
    """Raised when a nonlinear-reduction (softmax/layernorm) axis is assigned to spatial unrolling."""


class ReductionUnrollError(ValueError):
    """Raised when a reduction axis is spatially unrolled onto cores that cannot accumulate partial sums."""


def _as_dim_plus_const(expr: AffineExpr) -> tuple[int, int] | None:
    """Return ``(position, offset)`` if ``expr`` is exactly ``d`` or ``d + c`` (coefficient 1); else None."""
    if isinstance(expr, AffineDimExpr):
        return expr.position, 0
    if isinstance(expr, AffineBinaryOpExpr) and expr.kind == AffineBinaryOpKind.Add:
        for maybe_dim, maybe_const in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            if isinstance(maybe_dim, AffineDimExpr) and isinstance(maybe_const, AffineConstantExpr):
                return maybe_dim.position, maybe_const.value
    return None


def _self_offsets(affine_map: AffineMap) -> dict[int, int]:
    """Dimensions this map indexes as ``d + c`` with ``c != 0`` (a cross-iteration self-offset)."""
    offsets: dict[int, int] = {}
    for result in affine_map.results:
        parsed = _as_dim_plus_const(result)
        if parsed is not None and parsed[1] != 0:
            offsets[parsed[0]] = parsed[1]
    return offsets


def is_state_operand(node: HasIterationSpace, operand: Tensor) -> bool:
    """True when ``operand`` is a recurrence state input: read with a self-offset on a dimension the output writes
    without offset."""
    if operand not in node.inputs:
        return False
    offset_dims = set(_self_offsets(node.get_mapping(operand)))
    if not offset_dims:
        return False
    written = (
        set().union(*(map_dim_positions(node.get_mapping(out)) for out in node.outputs)) if node.outputs else set()
    )
    return bool(offset_dims & written)


def streamed_operands(node) -> tuple[Tensor, ...]:
    """The node's operands that travel, in the order a kernel declares them.

    A kernel's per-operand declarations -- layouts above all -- are read off by position, and
    a state operand occupies none: it is resident on the core rather than carried to it, so
    counting it would shift every operand after it, the output included.
    """
    inputs = tuple(getattr(node, "inputs", ()))
    outputs = tuple(getattr(node, "outputs", ()))
    if isinstance(node, HasIterationSpace):
        inputs = tuple(t for t in inputs if not is_state_operand(node, t))
    return (*inputs, *outputs)


def sequential_dims(node: HasIterationSpace) -> frozenset[int]:
    """Positions of the node's SEQUENTIAL iteration dimensions (cross-iteration state carry)."""
    written = (
        set().union(*(map_dim_positions(node.get_mapping(out)) for out in node.outputs)) if node.outputs else set()
    )
    sequential: set[int] = set()
    for operand in node.inputs:
        for position in _self_offsets(node.get_mapping(operand)):
            if position in written:
                sequential.add(position)
    return frozenset(sequential)


def nonlinear_reduction_dims(node: HasIterationSpace) -> frozenset[int]:
    """Positions this node reduces nonlinearly (softmax/layernorm); empty for an ordinary node."""
    declared = getattr(node, "reduction_axes", ())
    if declared:
        return frozenset(declared)
    if getattr(node, "fused_kernel", None) is None:
        return frozenset()
    return frozenset(pos for pos, kind in derive_iterator_types(node).items() if kind is IteratorType.REDUCTION)


def derive_iterator_types(node: HasIterationSpace) -> dict[int, IteratorType]:
    """Algorithmic type of every iteration dimension, keyed by position."""
    sequential = sequential_dims(node)
    output_dims = map_dim_positions(node.get_mapping(node.outputs[-1])) if node.outputs else frozenset()
    types: dict[int, IteratorType] = {}
    for position in range(node.num_dims):
        if position in sequential:
            types[position] = IteratorType.SEQUENTIAL
        elif position in output_dims:
            types[position] = IteratorType.PARALLEL
        else:
            types[position] = IteratorType.REDUCTION
    return types


def check_spatial_unroll_legal(node: HasIterationSpace, spatial_positions: Iterable[int]) -> None:
    """Raise if any spatially-unrolled dimension is SEQUENTIAL or a nonlinear (normalization) reduction."""
    positions = set(spatial_positions)
    illegal_sequential = sequential_dims(node) & positions
    if illegal_sequential:
        raise SequentialUnrollError(
            f"Node {node.name!r} dimension(s) {sorted(illegal_sequential)} carry a recurrent state "
            f"(SEQUENTIAL) and cannot be spatially unrolled; tile them temporally (chunk) instead."
        )
    illegal_nonlinear = nonlinear_reduction_dims(node) & positions
    if illegal_nonlinear:
        raise NonlinearReductionUnrollError(
            f"Node {node.name!r} dimension(s) {sorted(illegal_nonlinear)} are a nonlinear reduction "
            f"(softmax/layernorm) and cannot be spatially unrolled; fuse via the online-softmax rewrite "
            f"or keep the reduced axis resident."
        )


def check_spatial_unroll_accumulation_free(node: HasIterationSpace, spatial_positions: Iterable[int]) -> None:
    """Raise if a spatially-unrolled dimension is a REDUCTION.

    Only for backends without cross-core accumulation: they join cores by selecting one
    contribution, so the other cores' partial sums are silently dropped.
    """
    types = derive_iterator_types(node)
    illegal = sorted(p for p in set(spatial_positions) if types.get(p) is IteratorType.REDUCTION)
    if illegal:
        raise ReductionUnrollError(
            f"Node {node.name!r} dimension(s) {illegal} are a reduction and cannot be spatially unrolled "
            f"for a code-generated node: the backend has no cross-core accumulation, so the partial sums "
            f"would be dropped; unroll a parallel dimension or tile the reduction temporally instead."
        )
