"""Auto-propose fusion regions by greedy chain growth under a near-memory budget.

Given a workload and a near-memory capacity (in elements), this proposes which computation nodes to
fuse into one depth-first region. It is built entirely on the Phase 2 affine analysis, so it is *sound*
by construction:

- **Legality** comes from :func:`~stream.workload.fusion.analysis.edge_fusions`: a data-dependent read
  (MoE dispatch/combine, gather) is a hard barrier that can never sit inside a region.
- **Buffer** per edge is the streaming footprint from :func:`~stream.workload.fusion.analysis.pairwise_fusion`
  (the line-buffer/halo for a conv, the full row for a softmax reduction, O(1) for a recurrence state
  carry). A region's peak buffer is the max over its internal edges (only adjacent buffers are live in a
  streamed pipeline).
- **Recurrence chains are prioritized** without any op-specific code: chains grow in dataflow order and
  capacity is the only thing that cuts, so an O(1)-state carry always fits -- a recurrence chain stays
  fused even under a tiny capacity where a large materialized tensor would be cut.

This is the framework; an out-of-tree package can supply a calibrated buffer/capacity model through the
same interface. Nothing here imports a frontend or a concrete cost backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir.affine import AffineDimExpr

from stream.workload.affine_access import map_dim_positions
from stream.workload.fusion.analysis import consumer_reduction_axes, edge_fusions, pairwise_fusion
from stream.workload.iterator_type import IteratorType, derive_iterator_types, sequential_dims
from stream.workload.node import HasIterationSpace
from stream.workload.tensor import Tensor

__all__ = ["FusionRegion", "iteration_extents", "edge_stream_buffer", "propose_fusion_regions"]


@dataclass(frozen=True)
class FusionRegion:
    """A proposed depth-first fusion region."""

    nodes: tuple[str, ...]  # member computation-node names, in topological order
    buffer_elements: int  # peak inter-node buffer (max over internal edges); 0 for a lone node
    is_recurrence: bool  # contains a recurrence (SEQUENTIAL) carry -- streams with O(1) state
    boundary_reason: str  # why this region did not grow further: "data_dependent" | "capacity" | "sink"


def iteration_extents(node: HasIterationSpace) -> dict[int, int]:
    """Size of each iteration dimension, read off wherever it indexes an operand as a bare dim."""
    sizes: dict[int, int] = {}
    for tensor in node.tensors:
        for result, extent in zip(node.get_mapping(tensor).results, tensor.shape, strict=True):
            if isinstance(result, AffineDimExpr):
                sizes[result.position] = extent
    return sizes


def _candidate_fusion_dims(consumer: HasIterationSpace, tensor: Tensor) -> list[int]:
    """Consumer PARALLEL dimensions that index ``tensor`` and are not one of its reduction axes -- the
    axes a depth-first schedule could stream along."""
    types = derive_iterator_types(consumer)
    reductions = set(consumer_reduction_axes(consumer, tensor))
    # Use map_dim_positions so a dim inside a compound access (a conv's stride*oy+fy) still counts.
    indexed = map_dim_positions(consumer.get_mapping(tensor))
    return sorted(d for d in indexed if types.get(d) == IteratorType.PARALLEL and d not in reductions)


def edge_stream_buffer(producer: HasIterationSpace, consumer: HasIterationSpace, tensor: Tensor) -> int:
    """Smallest streaming buffer over the candidate fusion dimensions (the best depth-first fusion).

    Falls back to the whole shared-tensor size when no dimension streams cleanly (or the affine box
    cannot represent the access) -- a sound upper bound.
    """
    full = 1
    for size in tensor.shape:
        full *= size
    extents = iteration_extents(consumer)
    best = full
    for fusion_dim in _candidate_fusion_dims(consumer, tensor):
        if fusion_dim not in extents:
            continue
        try:
            window = pairwise_fusion(producer, consumer, fusion_dim, extents, tensor=tensor)
        except (NotImplementedError, ValueError, KeyError):
            continue
        best = min(best, window.buffer_elements)
    return best


def _collect_edges(workload, order: dict[str, int]) -> tuple[list[tuple[int, str, str]], set[tuple[str, str]]]:
    """Fusible compute-to-compute edges (buffer, producer, consumer) and the data-dependent barrier pairs.

    An edge may share more than one tensor (e.g. a flash-attention block carries both the output and the
    softmax-denominator state): the edge is a hard barrier if *any* shared read is data-dependent, else
    its buffer is the sum over the shared tensors (all are held live)."""
    edges: list[tuple[int, str, str]] = []
    barrier_between: set[tuple[str, str]] = set()
    for producer, consumer in workload.edges:
        if not (isinstance(producer, HasIterationSpace) and isinstance(consumer, HasIterationSpace)):
            continue
        if not (producer.name in order and consumer.name in order):
            continue
        shared = [t for t in producer.outputs if t in consumer.inputs]
        if not shared:
            continue
        if any(e.data_dependent for e in edge_fusions(producer, consumer)):
            barrier_between.add((producer.name, consumer.name))
            continue
        buffer = sum(edge_stream_buffer(producer, consumer, tensor) for tensor in shared)
        edges.append((buffer, producer.name, consumer.name))
    return edges, barrier_between


def _boundary_reason(members: set[str], barriers: set[tuple[str, str]], has_buffer: bool, is_whole: bool) -> str:
    """Why a region stopped growing: a data-dependent barrier out of it, else capacity, else the sink."""
    if any(prod in members and cons not in members for prod, cons in barriers):
        return "data_dependent"
    return "capacity" if has_buffer and not is_whole else "sink"


def propose_fusion_regions(workload, capacity_elements: int) -> list[FusionRegion]:
    """Greedy chain growth: grow depth-first regions in dataflow order while the region peak buffer stays
    within ``capacity_elements``; data-dependent edges are hard region boundaries.

    Returns one :class:`FusionRegion` per connected fused component, in topological order.
    """
    compute = list(workload.get_computation_nodes())
    order = {n.name: i for i, n in enumerate(compute)}
    edges, barrier_between = _collect_edges(workload, order)

    # Union-find over node names; a region's peak buffer is the max of its merged edge buffers.
    parent = {n.name: n.name for n in compute}
    region_buffer = {n.name: 0 for n in compute}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def merge_would_trap_a_barrier(root_a: str, root_b: str) -> bool:
        """Merging must not put both ends of a data-dependent edge in one region (that edge would then
        sit *inside* a fused region -- illegal). Catches e.g. MoE ``combine`` reachable from ``router``
        via a fusible edge while ``expert_out->combine`` is a hard barrier."""
        for prod, cons in barrier_between:
            if {find(prod), find(cons)} == {root_a, root_b}:
                return True
        return False

    # Grow chains in dataflow (topological-by-consumer) order; capacity is the only thing that cuts,
    # so an O(1)-state recurrence carry always fits (prioritized) while a large materialized tensor cuts.
    for buffer, prod, cons in sorted(edges, key=lambda e: (order[e[2]], order[e[1]])):
        ra, rb = find(prod), find(cons)
        if ra == rb:
            region_buffer[ra] = max(region_buffer[ra], buffer)
            continue
        if merge_would_trap_a_barrier(ra, rb):
            continue  # legality boundary: a data-dependent edge would end up inside the region
        merged_peak = max(region_buffer[ra], region_buffer[rb], buffer)
        if merged_peak > capacity_elements:
            continue  # capacity boundary: leave the two regions separate
        parent[rb] = ra
        region_buffer[ra] = merged_peak

    # Group nodes by their region root, preserving topological order.
    groups: dict[str, list[str]] = {}
    for node in compute:
        groups.setdefault(find(node.name), []).append(node.name)

    node_by_name = {n.name: n for n in compute}
    regions: list[FusionRegion] = []
    for root, names in sorted(groups.items(), key=lambda kv: order[kv[1][0]]):
        member_set = set(names)
        regions.append(
            FusionRegion(
                nodes=tuple(sorted(names, key=lambda n: order[n])),
                buffer_elements=region_buffer[root],
                is_recurrence=any(sequential_dims(node_by_name[n]) for n in names),
                boundary_reason=_boundary_reason(
                    member_set, barrier_between, region_buffer[root] > 0, len(names) == len(compute)
                ),
            )
        )
    return regions
