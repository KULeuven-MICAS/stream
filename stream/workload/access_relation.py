"""Typed operand access relation (plan/11).

``AccessRelation = AffineAccess | PiecewiseAffineAccess | DataDependentAccess`` -- a value object over
*one operand's* access, derived from the node's existing xDSL operand map. It does not replace
:attr:`~stream.workload.node.HasIterationSpace.operand_mapping` (still ``tuple[AffineMap, ...]``) and
adds no node field: :func:`access_for` returns an :class:`AffineAccess` for every node built today, so
the default derivation is affine and all current behavior is bit-identical.

The point of the sum type is that the parts of a modern workload the affine box cannot express become a
*parameter*, never a blocker:

- :class:`AffineAccess` -- the default; derived relevancy + exact footprint (delegates to
  :mod:`stream.workload.affine_access`).
- :class:`PiecewiseAffineAccess` -- a union of affine pieces (mask/padding/guarded regions); footprint
  is the per-result hull, with the optional islpy path for the exact union.
- :class:`DataDependentAccess` -- a static bounding access plus a descriptor whose runtime-unknown part
  (gather indices, dispatch table) is lifted into a swept-or-calibrated ``reuse`` parameter.
"""

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
    "register_data_dependent_op",
    "is_data_dependent_op",
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
        """False only for :class:`DataDependentAccess` -- the one kind fusion must treat as a barrier
        unless a rewrite lifts it. Affine and piecewise-affine access are static."""

    @abstractmethod
    def indexed_dims(self) -> frozenset[int]:
        """Iteration-dimension positions that index this operand (the VARYING dimensions)."""

    @abstractmethod
    def footprint(self, tile: Tile) -> tuple[range, ...]:
        """Per-result index footprint of an iteration ``tile`` for this operand."""

    def relevancy(self, dim: int | AffineDimExpr, num_dims: int) -> LoopEffect:
        """VARYING if ``dim`` indexes the operand, INVARIANT if it is a node dimension that does not,
        ABSENT if it is outside the node's ``num_dims`` -- identical to
        :func:`stream.workload.affine_access.relevancy`."""
        pos = _position(dim)
        if pos < 0 or pos >= num_dims:
            return LoopEffect.ABSENT
        return LoopEffect.VARYING if pos in self.indexed_dims() else LoopEffect.INVARIANT


@dataclass(frozen=True)
class AffineAccess(AccessRelation):
    """An operand indexed by a single affine map -- the default and the only kind any node builds today.

    Every query delegates to :mod:`stream.workload.affine_access`, so an ``AffineAccess`` wrapping
    ``node.get_mapping(operand)`` is bit-identical to the derived-access API.
    """

    map: AffineMap

    @property
    def is_static(self) -> bool:
        return True

    def indexed_dims(self) -> frozenset[int]:
        return map_dim_positions(self.map)

    def footprint(self, tile: Tile) -> tuple[range, ...]:
        return _affine_footprint(self.map, tile)

    def exact_footprint(self, tile: Tile) -> tuple[range, ...]:
        """The islpy-exact footprint. Imported lazily so ``islpy`` stays an optional extra."""
        from stream.workload.affine_exact import exact_footprint  # noqa: PLC0415

        return exact_footprint(self.map, tile)


@dataclass(frozen=True)
class PiecewiseAffineAccess(AccessRelation):
    """A union of affine pieces -- e.g. a masked, padded, or otherwise guarded region that is a union
    of affine sub-accesses of one operand of the same rank.

    ``footprint`` is the per-result **hull** (bounding box) of the pieces' footprints; the exact union
    is the optional islpy path. Still static: the pieces are known ahead of time.
    """

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

    The runtime-unknown is lifted into parameters, never a blocker:

    - ``bounding`` -- the conservative static access (the "bounding shape"); ``footprint`` and
      ``indexed_dims`` fall back to it, so the op costs at worst-case unless calibrated. This is what
      :class:`~stream.parser.onnx.slice_gather.GatherParser` already does implicitly (full-axis read).
    - ``index_tensor`` -- the runtime index/descriptor tensor name.
    - ``reuse`` -- a calibration parameter (a reuse/locality fraction a consumer applies to the bounding
      cost) or the centre of a swept DSE bracket; ``None`` means pure worst-case.
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


# Op types whose *data* input is read with a runtime-dependent index (gather/scatter, MoE routing).
# Their affine map is a conservative bounding access (the whole indexed axis); the true selection is
# data-dependent. A private overlay adds its own routing ops via ``register_data_dependent_op`` rather
# than editing this set -- the plugin boundary for the DataDependent bucket.
_DATA_DEPENDENT_OPS: set[str] = {"Gather", "Scatter", "MoEDispatch", "MoECombine"}


def register_data_dependent_op(op_type: str) -> None:
    """Declare an op type whose first input is a data-dependent (gather-like) read, so
    :func:`access_for` returns a :class:`DataDependentAccess` for it."""
    _DATA_DEPENDENT_OPS.add(op_type)


def is_data_dependent_op(op_type: str) -> bool:
    """Whether ``op_type`` reads its data input with a runtime-dependent index."""
    return op_type in _DATA_DEPENDENT_OPS


def access_for(node: HasIterationSpace, operand: Tensor) -> AccessRelation:
    """The access relation of ``operand`` on ``node``.

    Affine by default -- the derivation is bit-identical for the whole affine family (Conv/Gemm/MatMul/
    Slice/Scan/elementwise/normalization) and every node built before this taxonomy existed. The one
    upgrade: the *data* input (``inputs[0]``) of a data-dependent op (:func:`is_data_dependent_op` --
    Gather, MoE dispatch/combine, …) becomes a :class:`DataDependentAccess` whose bounding access is the
    conservative affine map and whose ``index_tensor`` is the routing/index operand (if any). Footprint
    and relevancy are unchanged (they fall back to the bounding map); only ``is_static`` flips, which is
    exactly what fusion analysis must see. Frontends / a private overlay can also build
    :class:`PiecewiseAffineAccess` (masked/windowed regions) explicitly.
    """
    affine = AffineAccess(node.get_mapping(operand))
    op_type = getattr(node, "type", None)
    if op_type is not None and op_type in _DATA_DEPENDENT_OPS and node.inputs and operand == node.inputs[0]:
        index_tensor = node.inputs[-1].name if len(node.inputs) > 1 else ""
        return DataDependentAccess(bounding=affine, index_tensor=index_tensor)
    return affine
