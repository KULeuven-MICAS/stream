"""Typed operand access relation over one operand: ``AffineAccess | PiecewiseAffineAccess |
DataDependentAccess``, derived from the node's xDSL operand map (affine by default, so behavior is unchanged)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.workload.affine_access import footprint as _affine_footprint
from stream.workload.affine_access import map_dim_positions
from stream.workload.node import HasIterationSpace
from stream.workload.steady_state.iteration_space import LoopEffect
from stream.workload.tensor import Tensor

__all__ = [
    "AccessRelation",
    "AffineAccess",
    "PiecewiseAffineAccess",
    "DataDependentAccess",
    "access_for",
]

Tile = Mapping[int | AffineDimExpr, range]
"""The (contiguous) extent of each iteration dimension a footprint query bounds."""


def _position(dim: int | AffineDimExpr) -> int:
    return dim.position if isinstance(dim, AffineDimExpr) else int(dim)


class AccessRelation(ABC):
    """How one operand's tensor indices depend on the node's iteration space."""

    @property
    @abstractmethod
    def is_static(self) -> bool:
        """False only for :class:`DataDependentAccess` (a fusion barrier); affine and piecewise are static."""

    @abstractmethod
    def indexed_dims(self) -> frozenset[int]:
        """Iteration-dimension positions that index this operand (the VARYING dimensions)."""

    @abstractmethod
    def footprint(self, tile: Tile) -> tuple[range, ...]:
        """Per-result index footprint of an iteration ``tile`` for this operand."""

    def relevancy(self, dim: int | AffineDimExpr, num_dims: int) -> LoopEffect:
        """VARYING if ``dim`` indexes the operand, INVARIANT if a node dim that does not index it,
        ABSENT if ``dim`` is outside the node's ``num_dims``."""
        pos = _position(dim)
        if pos < 0 or pos >= num_dims:
            return LoopEffect.ABSENT
        return LoopEffect.VARYING if pos in self.indexed_dims() else LoopEffect.INVARIANT


@dataclass(frozen=True)
class AffineAccess(AccessRelation):
    """An operand indexed by a single affine map; queries delegate to :mod:`stream.workload.affine_access`."""

    map: AffineMap

    @property
    def is_static(self) -> bool:
        return True

    def indexed_dims(self) -> frozenset[int]:
        return map_dim_positions(self.map)

    def footprint(self, tile: Tile) -> tuple[range, ...]:
        return _affine_footprint(self.map, tile)


@dataclass(frozen=True)
class PiecewiseAffineAccess(AccessRelation):
    """A union of affine pieces (masked/padded/guarded regions of one operand, same rank). ``footprint``
    is the per-result hull of the pieces; still static."""

    pieces: tuple[AffineAccess, ...]

    def __post_init__(self) -> None:
        if not self.pieces:
            raise ValueError("PiecewiseAffineAccess requires at least one affine piece")
        ranks = {len(piece.map.results) for piece in self.pieces}
        if len(ranks) != 1:
            raise ValueError(f"all pieces must have the same operand rank; got {sorted(ranks)}")

    @property
    def is_static(self) -> bool:
        return True

    def indexed_dims(self) -> frozenset[int]:
        dims: frozenset[int] = frozenset()
        for piece in self.pieces:
            dims |= piece.indexed_dims()
        return dims

    def footprint(self, tile: Tile) -> tuple[range, ...]:
        prints = [piece.footprint(tile) for piece in self.pieces]
        rank = len(prints[0])
        return tuple(range(min(fp[i].start for fp in prints), max(fp[i].stop for fp in prints)) for i in range(rank))


@dataclass(frozen=True)
class DataDependentAccess(AccessRelation):
    """An operand whose index set depends on runtime data (gather/scatter, MoE dispatch, masks).

    ``bounding`` is the conservative static access that ``footprint``/``indexed_dims`` fall back to;
    ``index_tensor`` names the runtime index tensor; ``reuse`` is an optional locality/DSE fraction
    (``None`` = worst case).
    """

    bounding: AffineAccess
    index_tensor: str
    reuse: float | None = None

    @property
    def is_static(self) -> bool:
        return False

    def indexed_dims(self) -> frozenset[int]:
        return self.bounding.indexed_dims()

    def footprint(self, tile: Tile) -> tuple[range, ...]:
        return self.bounding.footprint(tile)


# Op types whose data input is read with a runtime-dependent index (gather/scatter, MoE routing); their
# affine map is a conservative bounding access (the whole indexed axis).
_DATA_DEPENDENT_OPS: set[str] = {"Gather", "Scatter", "MoEDispatch", "MoECombine"}


def access_for(node: HasIterationSpace, operand: Tensor) -> AccessRelation:
    """Access relation of ``operand`` on ``node``: affine by default (bit-identical), except the data
    input of a data-dependent op (Gather, MoE dispatch/combine) becomes a :class:`DataDependentAccess`
    -- same footprint/relevancy, only ``is_static`` flips (what fusion analysis reads)."""
    affine = AffineAccess(node.get_mapping(operand))
    op_type = getattr(node, "type", None)
    if op_type is not None and op_type in _DATA_DEPENDENT_OPS and node.inputs and operand == node.inputs[0]:
        index_tensor = node.inputs[-1].name if len(node.inputs) > 1 else ""
        return DataDependentAccess(bounding=affine, index_tensor=index_tensor)
    return affine
