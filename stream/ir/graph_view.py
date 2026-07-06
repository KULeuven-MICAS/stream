"""WorkloadGraphView -- one uniform, smart graph view of any workload (tiled or untiled).

A single Stream API that Vortex (and any other consumer) renders directly: a proper node/edge graph
plus the two structural lenses that make a large workload legible --

- **Repeated-block collapse** (M05): :func:`~stream.workload.structure.find_repeated_blocks` groups
  computation nodes that are the same computation in the same structural position. Each such node
  carries a ``block_class`` id and the view lists the classes (representative + multiplicity +
  members), so a consumer can draw one representative and mark the rest ``×N`` -- clearly duplicated,
  cleanly skippable -- while the edges stay intact.
- **Fusable regions** (zoom): :meth:`~stream.workload.workload.Workload.split_fusion_groups` cuts the
  graph at layout barriers; each node carries its ``region`` id so a consumer can zoom into one
  fusible region at a time.

Every node also carries its *derived* affine metadata (iterator roles, data reuse, data movement,
normalization decomposition, recurrence) so the same view drives the affine lens uniformly. Nothing
here is workload-type specific -- it reads a ``Workload``, so a raw graph, a chunked decomposition, or
a tiled steady-state graph all serialize the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from stream.workload.affine_access import footprint, map_dim_positions
from stream.workload.fusion.proposer import propose_fusion_regions
from stream.workload.iterator_type import derive_iterator_types, sequential_dims
from stream.workload.node import (
    ComputationNode,
    FusionEdge,
    HasInputs,
    HasIterationSpace,
    HasOutputs,
    InEdge,
    NormalizationNode,
    OutEdge,
    TransferNode,
)
from stream.workload.normalization import decompose_normalization, parallel_axes
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.structure import find_repeated_blocks

if TYPE_CHECKING:
    from stream.workload.workload import Workload

_MOVEMENT_TYPES = {"Slice", "Gather"}


class AxisRefIR(BaseModel):
    pos: int = Field(description="Iteration-dimension position")
    size: int | None = Field(default=None, description="Extent of the axis, if known")


class GraphDimIR(BaseModel):
    name: str = Field(description="Loop-dimension name")
    size: int | None = Field(default=None)
    iterator_type: str = Field(description="PARALLEL | REDUCTION | SEQUENTIAL")


class OperandReuseIR(BaseModel):
    operand: str = Field(description="Input tensor reused across an output axis")
    axes: list[AxisRefIR] = Field(description="Output axes this input is INVARIANT over (reused across)")


class MovedRangeIR(BaseModel):
    axis: int
    read: tuple[int, int] = Field(description="[lo, hi) source range actually read")
    full: int = Field(description="Full extent of the source axis")


class MovementIR(BaseModel):
    kind: str = Field(description="'slice' (exact) or 'gather' (conservative)")
    source: str = Field(description="Source tensor moved from")
    moved: list[MovedRangeIR] = Field(default_factory=list)
    conservative: bool = Field(description="True when the whole source axis is the safe bound (gather)")


class SubOpIR(BaseModel):
    name: str
    op: str
    reduces: bool = Field(description="A reduction sub-op (contracts the normalized axis)")
    dims: list[GraphDimIR] = Field(
        default_factory=list,
        description="This sub-op's loop dims, typed REDUCTION (contracted here) or PARALLEL (broadcast) "
        "-- the dimension propagation through the decomposition, mapped back to the parent's named axes",
    )


class DecompositionIR(BaseModel):
    """The affine sub-operator graph a normalization expands into (max→exp→sum→div, …), with each
    sub-op's dimension roles, so a consumer can render the internal dataflow and inspect the axis
    propagation (which axis is reduced where, which is broadcast back)."""

    nodes: list[SubOpIR]
    edges: list[dict[str, str]]
    entry: list[str] = Field(
        default_factory=list, description="Sub-ops that read the normalization's input (wire producers here)"
    )
    exit: list[str] = Field(
        default_factory=list, description="Sub-ops that produce the normalization's output (wire consumers here)"
    )


class GraphNodeIR(BaseModel):
    name: str
    op: str = Field(description="Operator/role label, e.g. 'MatMul', 'Softmax', 'Slice', 'Transpose'")
    kind: str = Field(description="compute | normalization | data_movement | barrier | transfer | input | output")
    dims: list[GraphDimIR] = Field(default_factory=list)
    is_recurrence: bool = False
    block_class: int | None = Field(default=None, description="Repeated-block class id (None = unique)")
    region: int | None = Field(default=None, description="Fusable-region id (barrier-cut)")
    proposed_region: int | None = Field(default=None, description="Auto-proposed fusion-region id")
    reuse: list[OperandReuseIR] = Field(default_factory=list)
    reduction_axes: list[AxisRefIR] | None = None
    parallel_axes: list[AxisRefIR] | None = None
    movement: MovementIR | None = None
    decomposition: DecompositionIR | None = None


class GraphEdgeIR(BaseModel):
    source: str
    target: str
    shared_tensors: list[str] = Field(default_factory=list)


class BlockClassIR(BaseModel):
    id: int
    representative: str = Field(description="One member drawn to stand for the whole class")
    op: str
    multiplicity: int
    members: list[str]


class RegionIR(BaseModel):
    id: int
    nodes: list[str] = Field(description="Computation-node names in this fusable region")


class ProposedRegionIR(BaseModel):
    """A region the auto-proposer suggests fusing: greedy dataflow chain growth under a near-memory
    capacity, from the same affine analysis. Legal by construction (no data-dependent read inside)."""

    id: int
    nodes: list[str] = Field(description="Computation-node names fused into this region (topological)")
    buffer_elements: int = Field(description="Peak inter-node buffer the fusion holds (elements)")
    is_recurrence: bool = Field(description="Contains a recurrence state carry -- streams with O(1) state")
    boundary_reason: str = Field(description="Why it stops: 'data_dependent' | 'capacity' | 'sink'")


class WorkloadGraphView(BaseModel):
    """Uniform graph view of a workload: proper nodes+edges, repeated-block collapse, fusable regions.

    Construction is always via :meth:`from_workload`.
    """

    model_config = ConfigDict(
        json_schema_extra={"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "stream/workload_graph/v1"}
    )

    schema_version: Literal["1.0"] = "1.0"
    tiled: bool = Field(description="True if the workload is a tiled/steady-state graph")
    nodes: list[GraphNodeIR]
    edges: list[GraphEdgeIR]
    block_classes: list[BlockClassIR] = Field(description="Repeated-block classes, most-repeated first")
    regions: list[RegionIR] = Field(description="Fusable regions the barriers cut the graph into")
    proposed_regions: list[ProposedRegionIR] = Field(
        default_factory=list, description="Auto-proposed fusion regions (empty unless a capacity is given)"
    )

    @classmethod
    def from_workload(cls, workload: Workload, fusion_capacity: int | None = None) -> WorkloadGraphView:
        block_of, block_classes = _block_structure(workload)
        region_of, regions = _region_structure(workload)
        proposed_of, proposed_regions = _proposed_structure(workload, fusion_capacity)
        order = _topo_names(workload)
        by_name = {n.name: n for n in workload.nodes}
        nodes = [_node_ir(workload, by_name[name], block_of, region_of, proposed_of) for name in order]
        edges = [GraphEdgeIR(source=s.name, target=t.name, shared_tensors=_shared(s, t)) for s, t in workload.edges]
        return cls(
            tiled=_is_tiled(workload),
            nodes=nodes,
            edges=edges,
            block_classes=block_classes,
            regions=regions,
            proposed_regions=proposed_regions,
        )


# --------------------------------------------------------------------------- structure


def _is_tiled(workload: Workload) -> bool:
    return any(isinstance(n, SteadyStateComputation) for n in workload.nodes)


def _topo_names(workload: Workload) -> list[str]:
    return [n.name for n in nx.lexicographical_topological_sort(workload, key=lambda n: n.name)]


def _shared(src, dst) -> list[str]:
    if isinstance(src, HasOutputs) and isinstance(dst, HasInputs):
        return [t.name for t in src.outputs if t in dst.inputs]
    return []


def _block_structure(workload: Workload) -> tuple[dict[str, int], list[BlockClassIR]]:
    classes = find_repeated_blocks(workload)
    block_of: dict[str, int] = {}
    block_classes: list[BlockClassIR] = []
    for class_id, block in enumerate(classes):
        for node in block.nodes:
            block_of[node.name] = class_id
        block_classes.append(
            BlockClassIR(
                id=class_id,
                representative=block.nodes[0].name,
                op=block.nodes[0].type,
                multiplicity=block.multiplicity,
                members=[n.name for n in block.nodes],
            )
        )
    return block_of, block_classes


def _region_structure(workload: Workload) -> tuple[dict[str, int], list[RegionIR]]:
    try:
        groups = workload.split_fusion_groups()
    except Exception:  # noqa: BLE001 -- a graph the coarse splitter can't cut is reported as one region
        groups = [workload]
    region_of: dict[str, int] = {}
    regions: list[RegionIR] = []
    for region_id, group in enumerate(groups):
        names = [c.name for c in group.get_computation_nodes()]
        for name in names:
            region_of[name] = region_id
        regions.append(RegionIR(id=region_id, nodes=names))
    return region_of, regions


def _proposed_structure(workload: Workload, capacity: int | None) -> tuple[dict[str, int], list[ProposedRegionIR]]:
    """Auto-proposed fusion regions at the given near-memory ``capacity`` (in elements). Empty when no
    capacity is requested, so existing callers are unchanged."""
    if capacity is None:
        return {}, []
    proposed_of: dict[str, int] = {}
    regions: list[ProposedRegionIR] = []
    for region_id, region in enumerate(propose_fusion_regions(workload, capacity)):
        for name in region.nodes:
            proposed_of[name] = region_id
        regions.append(
            ProposedRegionIR(
                id=region_id,
                nodes=list(region.nodes),
                buffer_elements=region.buffer_elements,
                is_recurrence=region.is_recurrence,
                boundary_reason=region.boundary_reason,
            )
        )
    return proposed_of, regions


# --------------------------------------------------------------------------- per-node


# Checked in order; NormalizationNode (a ComputationNode subclass) precedes the compute fallback.
_KIND_BY_CLASS: tuple[tuple[type, str], ...] = (
    (InEdge, "input"),
    (OutEdge, "output"),
    (FusionEdge, "barrier"),
    (TransferNode, "transfer"),
    (NormalizationNode, "normalization"),
)


def _kind(node) -> str:
    for cls, kind in _KIND_BY_CLASS:
        if isinstance(node, cls):
            return kind
    if isinstance(node, ComputationNode) and node.type in _MOVEMENT_TYPES:
        return "data_movement"
    return "compute"


def _op(node) -> str:
    if isinstance(node, ComputationNode):
        return node.type
    if isinstance(node, FusionEdge):
        return node.op_type
    return type(node).__name__


def _node_ir(
    workload: Workload, node, block_of: dict[str, int], region_of: dict[str, int], proposed_of: dict[str, int]
) -> GraphNodeIR:
    ir = GraphNodeIR(
        name=node.name,
        op=_op(node),
        kind=_kind(node),
        block_class=block_of.get(node.name),
        region=region_of.get(node.name),
        proposed_region=proposed_of.get(node.name),
    )
    if not isinstance(node, HasIterationSpace):
        return ir

    dims = workload.get_dims(node)
    iterator_types = derive_iterator_types(node)
    ir.dims = [
        GraphDimIR(name=str(d), size=_size(workload, d), iterator_type=iterator_types[p].name)
        for p, d in enumerate(dims)
    ]
    ir.is_recurrence = bool(sequential_dims(node))
    ir.reuse = _reuse(node, ir.dims)

    if isinstance(node, NormalizationNode):
        _annotate_normalization(ir, node)
    if isinstance(node, ComputationNode) and node.type in _MOVEMENT_TYPES:
        ir.movement = _movement(node)
    return ir


def _size(workload: Workload, dim) -> int | None:
    try:
        return workload.get_dimension_size(dim)
    except Exception:  # noqa: BLE001 -- unknown sizes render as blank, not a crash
        return None


def _reuse(node: ComputationNode, dims: list[GraphDimIR]) -> list[OperandReuseIR]:
    """Inputs held INVARIANT across a high-extent output axis -- the affine data-reuse statement
    (GQA's K/V reused across query heads; any matmul's weight reused across rows)."""
    if not node.outputs:
        return []
    out_positions = map_dim_positions(node.get_mapping(node.outputs[-1]))
    reuse: list[OperandReuseIR] = []
    for operand in node.inputs:
        indexed = map_dim_positions(node.get_mapping(operand))
        axes = [
            AxisRefIR(pos=p, size=dims[p].size)
            for p in sorted(out_positions - indexed)
            if p < len(dims) and (dims[p].size or 0) > 1
        ]
        if axes:
            reuse.append(OperandReuseIR(operand=operand.name, axes=axes))
    return reuse


def _subop_dims(sub_node: ComputationNode, parent_dims: list[GraphDimIR], reduced: tuple[int, ...]) -> list[GraphDimIR]:
    """Map a decomposition sub-op's iteration axes back onto the parent normalization's named dims, so
    each sub-op shows which axes it iterates and whether each is REDUCTION (contracted here) or PARALLEL
    (broadcast). A full-rank sub-op (max/exp/sum/div) aligns 1:1 with the parent; a reduced-rank sub-op
    (e.g. LpNorm's Sqrt over the statistic) iterates only the kept, non-reduced axes."""
    iterator_types = derive_iterator_types(sub_node)
    rank = len(iterator_types)
    positions = (
        list(range(rank)) if rank == len(parent_dims) else [p for p in range(len(parent_dims)) if p not in set(reduced)]
    )
    dims: list[GraphDimIR] = []
    for i, parent_pos in enumerate(positions):
        if i >= rank or parent_pos >= len(parent_dims):
            continue
        pd = parent_dims[parent_pos]
        dims.append(GraphDimIR(name=pd.name, size=pd.size, iterator_type=iterator_types[i].name))
    return dims


def _annotate_normalization(ir: GraphNodeIR, node: NormalizationNode) -> None:
    reduced = set(node.reduction_axes)
    for pos, dim in enumerate(ir.dims):
        if pos in reduced:
            dim.iterator_type = "REDUCTION"
    ir.reduction_axes = [AxisRefIR(pos=p, size=ir.dims[p].size) for p in node.reduction_axes if p < len(ir.dims)]
    ir.parallel_axes = [AxisRefIR(pos=p, size=ir.dims[p].size) for p in parallel_axes(node) if p < len(ir.dims)]
    if node.reduction_axes:
        try:
            dec = decompose_normalization(node)
        except NotImplementedError:
            return
        sub = dec.get_computation_nodes()
        sub_set = set(sub)
        ir.decomposition = DecompositionIR(
            nodes=[
                SubOpIR(
                    name=n.name,
                    op=n.type,
                    reduces=n.type.startswith("Reduce"),
                    dims=_subop_dims(n, ir.dims, node.reduction_axes),
                )
                for n in sub
            ],
            edges=[{"source": s.name, "target": t.name} for s, t in dec.edges if s in sub_set and t in sub_set],
            entry=[t.name for s, t in dec.edges if isinstance(s, InEdge) and t in sub_set],
            exit=[s.name for s, t in dec.edges if isinstance(t, OutEdge) and s in sub_set],
        )


def _movement(node: ComputationNode) -> MovementIR:
    src, out = node.inputs[0], node.outputs[0]
    if node.type == "Slice":
        tile = {pos: range(0, size) for pos, size in enumerate(out.shape)}
        region = footprint(node.operand_mapping[0], tile)
        moved = [
            MovedRangeIR(axis=ax, read=(r.start, r.stop), full=src.shape[ax])
            for ax, r in enumerate(region)
            if (r.start, r.stop) != (0, src.shape[ax])
        ]
        return MovementIR(kind="slice", source=src.name, moved=moved, conservative=False)
    return MovementIR(kind="gather", source=src.name, moved=[], conservative=True)
