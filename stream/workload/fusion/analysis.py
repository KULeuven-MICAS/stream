"""Affine fusion analysis: fusibility and streaming buffer size from producer/consumer map composition."""

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
        """Streamable iff the window is a strict, forward-advancing sub-range of the full extent."""
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
    """A unit consumer tile: fusion dim pinned to ``[at, at+1)``, all other shared-tensor dims at full extent."""
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
    """Analyse streaming ``producer -> consumer`` fusion along one consumer ``fusion_dim``; ``tensor`` selects the
    shared tensor when several are shared (defaults to the unique one)."""
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
        # Consumer read doesn't advance with the fusion dim (e.g. full reduction): window == full, not streamable.
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


# AccessRelation-aware edge classification (data-dependent / reduction barriers).
@dataclass(frozen=True)
class EdgeFusion:
    """Fusibility of one producer->consumer edge from the consumer's AccessRelation on the shared tensor.

    A data-dependent read is a hard barrier; a normalization's reduced axis is a per-axis barrier.
    """

    producer: str
    consumer: str
    tensor: str
    data_dependent: bool
    reduction_axes: tuple[int, ...]
    nonlinear_reduction: bool = False

    @property
    def fusible(self) -> bool:
        """False only for a data-dependent (hard) barrier."""
        return not self.data_dependent


def _shared_tensors(producer: HasIterationSpace, consumer: HasIterationSpace) -> list[Tensor]:
    return [t for t in producer.outputs if t in consumer.inputs]


def consumer_reduction_axes(consumer: HasIterationSpace, tensor: Tensor) -> tuple[int, ...]:
    """Consumer dimensions that both reduce and index ``tensor`` -- the per-axis fusion barriers (other axes fuse
    freely)."""
    indexed = map_dim_positions(consumer.get_mapping(tensor))
    reducing = {pos for pos, it in derive_iterator_types(consumer).items() if it == IteratorType.REDUCTION}
    if isinstance(consumer, NormalizationNode):
        reducing.update(consumer.reduction_axes)
    return tuple(sorted(reducing & indexed))


def edge_fusions(producer: HasIterationSpace, consumer: HasIterationSpace) -> list[EdgeFusion]:
    """Classify every shared tensor on a producer->consumer edge; a nonlinear (normalization) reduction is flagged
    because streaming it needs the online-softmax rewrite."""
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
    """Classify every compute-to-compute edge of ``workload`` (skips InEdge/OutEdge/FusionEdge boundaries)."""
    edges: list[EdgeFusion] = []
    for producer, consumer in workload.edges:
        if isinstance(producer, HasIterationSpace) and isinstance(consumer, HasIterationSpace):
            edges.extend(edge_fusions(producer, consumer))
    return edges
