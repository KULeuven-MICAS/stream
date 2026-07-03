"""Affine fusion analysis: fusibility + streaming buffer size from map composition (plan/06).

For a producer P (output map on a shared tensor) and consumer Q (input map on the same tensor),
tiling a *fusion dimension* of Q depth-first lets P stream into Q if the producer region a consumer
tile needs is a **bounded window that advances monotonically** as the tile sweeps the fusion dim.
Everything is derived from the xDSL affine maps via M02's :func:`compose_dependency` / :func:`footprint`
-- no op-specific heuristics, so it generalises to new operators and recurrences.

Two proof points fall out with no special-casing:
- a SEQUENTIAL recurrence streams along its sequence axis with an O(1) state window (the state is read
  at ``t-1`` only), and
- softmax / any full-reduction consumer is *not* fusible along the reduced axis (the window spans the
  whole axis), which must be reported, never silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir.affine import AffineDimExpr

from stream.workload.affine_access import compose_dependency, footprint, map_dim_positions
from stream.workload.node import HasIterationSpace
from stream.workload.tensor import Tensor

__all__ = ["FusionWindow", "pairwise_fusion", "shared_tensor"]


@dataclass(frozen=True)
class FusionWindow:
    """Result of analysing one producer->consumer edge along one consumer fusion dimension."""

    fusion_dim: int
    window: int  # producer iterations kept live for a unit consumer tile (the buffer window)
    step: int  # how far that window advances per unit consumer step (monotone advance)
    full: int  # the producer's full extent along the fused axis (window == full ⇒ not streamable)
    buffer_elements: int  # shared-tensor elements the window spans (the inter-node buffer size)

    @property
    def fusible(self) -> bool:
        """Streamable along this dim iff the window is a strict, forward-advancing sub-range of the
        full producer extent (a bounded line buffer), not the whole axis."""
        return self.step > 0 and self.window < self.full

    @property
    def halo(self) -> int:
        """Overlap carried between consecutive tiles = window - step (0 for a clean sliding tile)."""
        return max(0, self.window - self.step)


def shared_tensor(producer: HasIterationSpace, consumer: HasIterationSpace) -> Tensor:
    """The tensor a producer writes and a consumer reads (the fusion edge). Raises if none/ambiguous."""
    shared = [t for t in producer.outputs if t in consumer.inputs]
    if len(shared) != 1:
        raise ValueError(f"expected exactly one shared tensor, found {len(shared)}")
    return shared[0]


def _consumer_tile(consumer: HasIterationSpace, tensor: Tensor, fusion_dim: int, at: int, extents: dict[int, int]):
    """A unit consumer tile: the fusion dim pinned to ``[at, at+1)``, every other dim that indexes the
    shared tensor at its full extent (the consumer reads all of those per fusion step)."""
    indexed = map_dim_positions(consumer.get_mapping(tensor))
    tile = {d: range(0, extents[d]) for d in indexed if d != fusion_dim}
    tile[fusion_dim] = range(at, at + 1)
    return tile


def pairwise_fusion(
    producer: HasIterationSpace,
    consumer: HasIterationSpace,
    fusion_dim: int,
    extents: dict[int, int],
) -> FusionWindow:
    """Analyse streaming ``producer -> consumer`` fusion along one consumer ``fusion_dim``.

    ``extents`` maps each consumer iteration-dimension position to its size. The window/step come from
    composing the consumer's read at two consecutive fusion positions back onto the producer's
    iteration space; ``buffer_elements`` is the shared-tensor footprint of that window.
    """
    tensor = shared_tensor(producer, consumer)
    p_out = producer.get_mapping(tensor)
    c_in = consumer.get_mapping(tensor)

    region0 = compose_dependency(p_out, c_in, _consumer_tile(consumer, tensor, fusion_dim, 0, extents))
    region1 = compose_dependency(p_out, c_in, _consumer_tile(consumer, tensor, fusion_dim, 1, extents))

    # Find the producer dim that carries the fusion (its region moves between position 0 and 1).
    moved = [d for d in region0 if d in region1 and region0[d] != region1[d]]
    if moved:
        d = moved[0]
        window = len(region0[d])
        step = region1[d].start - region0[d].start
        full = _producer_extent(p_out, d, tensor)
    else:
        # The consumer's read doesn't advance with the fusion dim (e.g. a full reduction): the whole
        # producer output is needed for every tile -> window == full, not streamable.
        d = next(iter(region0), fusion_dim)
        window = len(region0.get(d, range(0)))
        step = 0
        full = window

    # Buffer = shared-tensor elements the producer window spans.
    buffer_elements = _region_elements(footprint(p_out, region0))
    return FusionWindow(fusion_dim=fusion_dim, window=window, step=step, full=full, buffer_elements=buffer_elements)


def _producer_extent(p_out, dim: int, tensor: Tensor) -> int:
    """Full extent of the producer along iteration ``dim`` (from the shared tensor shape it writes)."""
    for result, size in zip(p_out.results, tensor.shape, strict=True):
        if isinstance(result, AffineDimExpr) and result.position == dim:
            return size
    return max(tensor.shape, default=1)


def _region_elements(ranges) -> int:
    total = 1
    for r in ranges:
        total *= max(1, len(r))
    return total
