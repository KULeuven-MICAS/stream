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

from stream.workload.access_relation import access_for
from stream.workload.affine_access import compose_dependency, footprint, map_dim_positions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import HasIterationSpace, NormalizationNode
from stream.workload.tensor import Tensor

__all__ = [
    "FusionWindow",
    "pairwise_fusion",
    "shared_tensor",
    "EdgeFusion",
    "edge_fusions",
    "workload_fusion_edges",
    "consumer_reduction_axes",
]


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
    tensor: Tensor | None = None,
) -> FusionWindow:
    """Analyse streaming ``producer -> consumer`` fusion along one consumer ``fusion_dim``.

    ``extents`` maps each consumer iteration-dimension position to its size. The window/step come from
    composing the consumer's read at two consecutive fusion positions back onto the producer's
    iteration space; ``buffer_elements`` is the shared-tensor footprint of that window. Pass ``tensor``
    to analyse a specific shared tensor when producer/consumer share more than one (e.g. a flash block
    carrying both the output and the softmax denominator); it defaults to the unique shared tensor.
    """
    if tensor is None:
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


# --------------------------------------------------------------------------- #
#  AccessRelation-aware edge classification (data-dependent / reduction barriers) #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdgeFusion:
    """Fusibility of one producer->consumer edge, classified from the consumer's AccessRelation on the
    shared tensor (the axis-level window/buffer is :func:`pairwise_fusion`; this is the barrier view).

    - ``data_dependent`` -- the consumer reads the shared tensor with a data-dependent index (gather /
      MoE dispatch or combine): a **hard barrier**, the producer cannot be fused in at all until a
      rewrite lifts the indexing. ``fusible`` is then False.
    - ``reduction_axes`` -- consumer axes that are a **full reduction** of the shared tensor (a
      normalization's reduced axis): fusible along the *other* (parallel) axes, a barrier along these.
    """

    producer: str
    consumer: str
    tensor: str
    data_dependent: bool
    reduction_axes: tuple[int, ...]
    nonlinear_reduction: bool = False

    @property
    def fusible(self) -> bool:
        """Fusible along at least the parallel axes. False only for a data-dependent (hard) barrier."""
        return not self.data_dependent


def _shared_tensors(producer: HasIterationSpace, consumer: HasIterationSpace) -> list[Tensor]:
    return [t for t in producer.outputs if t in consumer.inputs]


def consumer_reduction_axes(consumer: HasIterationSpace, tensor: Tensor) -> tuple[int, ...]:
    """Consumer iteration dimensions that both **reduce** and **index** ``tensor`` -- the axes along
    which fusing a producer through ``tensor`` needs the whole reduced slice (a per-axis barrier),
    while every other (parallel) axis fuses freely.

    Covers affine contractions (a MatMul's ``k``, a pool's window) via the derived ``REDUCTION``
    iterator type, **and** a normalization's ``reduction_axes`` (which its identity map hides from the
    derived types). This is what makes the attention key axis show up on *both* ``scores->softmax`` and
    ``softmax->context`` -- the single streaming axis flash attention rides.
    """
    indexed = map_dim_positions(consumer.get_mapping(tensor))
    reducing = {pos for pos, it in derive_iterator_types(consumer).items() if it == IteratorType.REDUCTION}
    if isinstance(consumer, NormalizationNode):
        reducing.update(consumer.reduction_axes)
    return tuple(sorted(reducing & indexed))


def edge_fusions(producer: HasIterationSpace, consumer: HasIterationSpace) -> list[EdgeFusion]:
    """Classify every shared tensor on a producer->consumer edge (usually one).

    A data-dependent consumer read is a hard barrier (``access_for`` reports a non-static access). Every
    reduction axis the consumer contracts over the shared tensor is a per-axis barrier
    (:func:`consumer_reduction_axes`); a **nonlinear** reduction (a normalization -- softmax) is flagged
    because streaming it needs the online-softmax rewrite, whereas a linear contraction streams with a
    plain accumulator. None of this uses op-specific code beyond the AccessRelation and the derived
    iterator types.
    """
    results: list[EdgeFusion] = []
    nonlinear = isinstance(consumer, NormalizationNode)
    for tensor in _shared_tensors(producer, consumer):
        data_dependent = not access_for(consumer, tensor).is_static
        reduction_axes = consumer_reduction_axes(consumer, tensor)
        results.append(
            EdgeFusion(
                producer=producer.name,
                consumer=consumer.name,
                tensor=tensor.name,
                data_dependent=data_dependent,
                reduction_axes=reduction_axes,
                nonlinear_reduction=nonlinear and bool(reduction_axes),
            )
        )
    return results


def workload_fusion_edges(workload) -> list[EdgeFusion]:
    """Classify every compute-to-compute edge of ``workload`` (skips InEdge/OutEdge/FusionEdge
    boundaries). The result is an annotative fusion map -- which producer->consumer edges are
    streamable, which are data-dependent hard barriers, and which carry a reduction axis."""
    edges: list[EdgeFusion] = []
    for producer, consumer in workload.edges:
        if isinstance(producer, HasIterationSpace) and isinstance(consumer, HasIterationSpace):
            edges.extend(edge_fusions(producer, consumer))
    return edges
