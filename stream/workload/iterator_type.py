"""Algorithmic iterator types, derived from the affine operand maps.

A node's iteration dimensions carry an *algorithmic* role, orthogonal to the mapping-placement
`IterationVariableType` (KERNEL/SPATIAL/TEMPORAL/...):

- ``PARALLEL``   -- indexes the output; freely tileable and spatially unrollable.
- ``REDUCTION``  -- an accumulation dim (indexes some input but not the output).
- ``SEQUENTIAL`` -- carries a cross-iteration state dependence (read at ``t-k``, written at ``t``);
                    a total order is required, so it must not be spatially unrolled.

Every classification is *derived* from the operand `AffineMap`s (derive, don't declare): a
recurrence is recognised by a state operand read with a constant self-offset on a dimension that the
output writes without offset. Conv/gemm padding (``oy + fy - 1``, two dims) is not a self-offset and
is never mistaken for a recurrence, so existing workloads classify exactly as PARALLEL/REDUCTION.
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
    "derive_iterator_types",
    "sequential_dims",
    "is_state_operand",
    "check_spatial_unroll_legal",
]


class IteratorType(Enum):
    PARALLEL = auto()
    REDUCTION = auto()
    SEQUENTIAL = auto()


class SequentialUnrollError(ValueError):
    """Raised when a SEQUENTIAL iteration dimension is assigned to spatial unrolling."""


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
    """True when ``operand`` is a recurrence state input: read with a self-offset on a dimension the
    output writes without offset (so the cross-iteration dependence distance is O(1))."""
    if operand not in node.inputs:
        return False
    offset_dims = set(_self_offsets(node.get_mapping(operand)))
    if not offset_dims:
        return False
    written = (
        set().union(*(map_dim_positions(node.get_mapping(out)) for out in node.outputs)) if node.outputs else set()
    )
    return bool(offset_dims & written)


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


def derive_iterator_types(node: HasIterationSpace) -> dict[int, IteratorType]:
    """Algorithmic type of every iteration dimension, keyed by position.

    SEQUENTIAL if it carries a state carry; else PARALLEL if it indexes the output (last output
    operand); else REDUCTION (accumulation dimension).
    """
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
    """Raise :class:`SequentialUnrollError` if any spatially-unrolled dimension is SEQUENTIAL.

    A recurrence's sequential dimension must keep a total order, so it can only be tiled temporally
    (chunked), never unrolled across space. No-op for nodes without a state carry.
    """
    illegal = sequential_dims(node) & set(spatial_positions)
    if illegal:
        raise SequentialUnrollError(
            f"Node {node.name!r} dimension(s) {sorted(illegal)} carry a recurrent state (SEQUENTIAL) "
            f"and cannot be spatially unrolled; tile them temporally (chunk) instead."
        )
