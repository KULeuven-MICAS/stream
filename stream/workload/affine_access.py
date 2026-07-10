"""Derived operand-access queries (relevancy, tile footprints, dependency regions) over a node's xDSL affine operand
maps."""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

from stream.workload.node import HasIterationSpace
from stream.workload.steady_state.iteration_space import LoopEffect
from stream.workload.tensor import Tensor

__all__ = [
    "Interval",
    "map_dim_positions",
    "relevancy",
    "operand_relevancy",
    "footprint",
    "compose_dependency",
]

Interval = tuple[int, int]
"""Inclusive ``(low, high)`` bounds of an integer quantity."""


def _position(dim: int | AffineDimExpr) -> int:
    if isinstance(dim, AffineDimExpr):
        return dim.position
    return int(dim)


def _const(expr: AffineExpr) -> int | None:
    return expr.value if isinstance(expr, AffineConstantExpr) else None


def _dims_in_expr(expr: AffineExpr) -> frozenset[int]:
    """Positions of every dimension that appears with a non-zero coefficient in ``expr``."""
    if isinstance(expr, AffineDimExpr):
        return frozenset({expr.position})
    if isinstance(expr, AffineBinaryOpExpr):
        return _dims_in_expr(expr.lhs) | _dims_in_expr(expr.rhs)
    return frozenset()


def map_dim_positions(affine_map: AffineMap) -> frozenset[int]:
    """Positions of all iteration dimensions that index any result of ``affine_map``."""
    positions: frozenset[int] = frozenset()
    for result in affine_map.results:
        positions |= _dims_in_expr(result)
    return positions


def relevancy(node: HasIterationSpace, operand: Tensor, dim: int | AffineDimExpr) -> LoopEffect:
    """How iteration dimension ``dim`` affects ``operand``: VARYING if it indexes the operand, INVARIANT if a node dim
    that does not, ABSENT if not a node dim."""
    pos = _position(dim)
    if pos < 0 or pos >= node.num_dims:
        return LoopEffect.ABSENT
    if pos in map_dim_positions(node.get_mapping(operand)):
        return LoopEffect.VARYING
    return LoopEffect.INVARIANT


def operand_relevancy(node: HasIterationSpace, operand: Tensor) -> dict[int, LoopEffect]:
    """Relevancy of every node iteration dimension for ``operand``, keyed by dimension position."""
    varying = map_dim_positions(node.get_mapping(operand))
    return {pos: (LoopEffect.VARYING if pos in varying else LoopEffect.INVARIANT) for pos in range(node.num_dims)}


def _range_bounds(extent: range) -> Interval:
    if len(extent) == 0:
        raise ValueError("tile range must be non-empty")
    return extent[0], extent[-1]


def _interval_of_expr(expr: AffineExpr, box: Mapping[int, Interval]) -> Interval:
    """Exact inclusive interval of an affine expression over an iteration box."""
    if isinstance(expr, AffineConstantExpr):
        return expr.value, expr.value
    if isinstance(expr, AffineDimExpr):
        if expr.position not in box:
            raise ValueError(f"tile does not bound dimension d{expr.position}")
        return box[expr.position]
    if isinstance(expr, AffineBinaryOpExpr):
        if expr.kind == AffineBinaryOpKind.Add:
            low_l, high_l = _interval_of_expr(expr.lhs, box)
            low_r, high_r = _interval_of_expr(expr.rhs, box)
            return low_l + low_r, high_l + high_r
        if expr.kind == AffineBinaryOpKind.Mul:
            coeff = _const(expr.lhs)
            other = expr.rhs
            if coeff is None:
                coeff = _const(expr.rhs)
                other = expr.lhs
            if coeff is None:
                raise NotImplementedError("product of two non-constant affine expressions is not box-representable")
            low, high = _interval_of_expr(other, box)
            return (coeff * low, coeff * high) if coeff >= 0 else (coeff * high, coeff * low)
        raise NotImplementedError(f"affine operator {expr.kind} has no exact box interval; use the islpy exact path")
    raise NotImplementedError(f"unsupported affine expression {type(expr).__name__}")


def _to_box(tile: Mapping[int | AffineDimExpr, range]) -> dict[int, Interval]:
    return {_position(dim): _range_bounds(extent) for dim, extent in tile.items()}


def footprint(affine_map: AffineMap, tile: Mapping[int | AffineDimExpr, range]) -> tuple[range, ...]:
    """Per-result index footprint (one contiguous ``range`` per operand index) of an iteration ``tile`` under
    ``affine_map``."""
    box = _to_box(tile)
    ranges: list[range] = []
    for result in affine_map.results:
        low, high = _interval_of_expr(result, box)
        ranges.append(range(low, high + 1))
    return tuple(ranges)


def compose_dependency(
    producer_out: AffineMap,
    consumer_in: AffineMap,
    consumer_tile: Mapping[int | AffineDimExpr, range],
) -> dict[int, range]:
    """Producer iteration region (``range`` per producer dim) needed for the shared-tensor slice a consumer tile reads;
    producer dims not indexing the shared tensor are omitted."""
    tensor_slice = footprint(consumer_in, consumer_tile)
    if len(tensor_slice) != len(producer_out.results):
        raise ValueError(
            f"shared tensor rank mismatch: producer produces {len(producer_out.results)} indices, "
            f"consumer reads {len(tensor_slice)}"
        )
    producer_region: dict[int, range] = {}
    for result, index_range in zip(producer_out.results, tensor_slice, strict=True):
        if not isinstance(result, AffineDimExpr):
            raise NotImplementedError(
                "compose_dependency requires a single-dimension (permutation) producer output map; "
                "got a composite expression. Use the islpy exact path."
            )
        producer_region[result.position] = index_range
    return producer_region
