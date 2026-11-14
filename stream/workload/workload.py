from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import combinations, pairwise, product
from math import ceil, gcd, prod
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import sympy as sp
from xdsl.dialects.memref import SubviewOp
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper

from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.memory import Memory
from stream.workload._svg import write_svg as _write_svg
from stream.workload.affine_access import footprint, map_dim_positions
from stream.workload.affine_transform import AffineTransform
from stream.workload.demand import Box, DemandItem, box_bits, less, meet, within
from stream.workload.iterator_type import IteratorType, derive_iterator_types, is_state_operand, sequential_dims
from stream.workload.node import (
    ComputationNode,
    FusionEdge,
    HasInputs,
    HasIterationSpace,
    HasOutputs,
    InEdge,
    Node,
    OutEdge,
    TransferNode,
)
from stream.workload.steady_state.iteration_space import IterationVariable, SteadyStateIterationSpace
from stream.workload.tensor import Tensor
from stream.workload.utils import affine_bounds, affine_coefficients, sympy_to_xdsl

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.mapping.mapping import EdgePlacement, Mapping


_SharedAxis = tuple[Node, int]
"""An axis of a tensor several nodes read: its producer, and the axis."""

_Read = tuple[Tensor, "Node", InterCoreTiling, int, int]
"""A copy's read at one target: the tensor read, the node reading it, that node's split, its core position and
the number of positions."""


@dataclass(frozen=True)
class _TargetDemand:
    """A box the ``copies`` read at ``target`` per firing, inside their ``window`` (each axis with whether a split
    places it); ``local_copy`` where the larger copy ``copied_from`` in the target's memory holds it."""

    target: int
    box: Box
    window: list[tuple[int, int, bool]]
    copies: tuple[int, ...]
    local_copy: bool
    copied_from: int | None = None


@dataclass(frozen=True)
class _FiringNest:
    """The fused loops over ``dims`` (innermost first, ``trips`` trips each) whose trips are the firings of a transfer
    whose copies slide together: per node reading a copy and per dim, the range of the dim it computes at each trip."""

    dims: tuple[LayerDim, ...]
    trips: tuple[int, ...]
    spans: dict["HasIterationSpace", dict[LayerDim, list[tuple[int, int]]]]

    @cached_property
    def trip_indices(self) -> list[tuple[int, ...]]:
        """Each firing's trip per dim, innermost first, the innermost dim fastest."""
        return [tuple(reversed(p)) for p in product(*(range(n) for n in reversed(self.trips)))]


_Firing = tuple[_FiringNest, int]
"""A firing: its nest and its index."""


def _disjoint_boxes(entries: list[_TargetDemand]) -> list[_TargetDemand]:
    """``entries`` recut into disjoint boxes, each with all the copies reading it, so that a part several copies read
    is one box, moved once for all of them."""
    if not entries:
        return []
    edges = [
        sorted({e for entry in entries for e in (entry.box[a][0], entry.box[a][1] + 1)})
        for a in range(len(entries[0].box))
    ]
    readers: dict[tuple[int, ...], tuple[set[int], set[int]]] = {}
    for entry in entries:
        ranges = [
            range(bisect_left(axis_edges, lo), bisect_left(axis_edges, hi + 1))
            for axis_edges, (lo, hi) in zip(edges, entry.box, strict=True)
        ]
        for cell in product(*ranges):
            copies, targets = readers.setdefault(cell, (set(), set()))
            copies.update(entry.copies)
            targets.add(entry.target)
    regions = {
        tuple((edges[a][k], edges[a][k + 1] - 1) for a, k in enumerate(cell)): (tuple(sorted(c)), tuple(sorted(t)))
        for cell, (c, t) in readers.items()
    }
    for axis in reversed(range(len(edges))):
        regions = _merged_along(regions, axis)
    return [
        _TargetDemand(target, box, _window_of(entries, target, box), copies, False)
        for box, (copies, targets) in regions.items()
        for target in targets
    ]


def _merged_along(
    regions: dict[Box, tuple[tuple[int, ...], tuple[int, ...]]], axis: int
) -> dict[Box, tuple[tuple[int, ...], tuple[int, ...]]]:
    """``regions`` with boxes of the same copies and targets that touch along ``axis`` merged into one."""
    rows: dict[tuple, list[Box]] = {}
    for box, key in regions.items():
        rows.setdefault((key, box[:axis], box[axis + 1 :]), []).append(box)
    merged: dict[Box, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for (key, _, _), boxes in rows.items():
        boxes.sort(key=lambda box: box[axis][0])
        run = boxes[0]
        for box in boxes[1:]:
            if box[axis][0] == run[axis][1] + 1:
                run = (*run[:axis], (run[axis][0], box[axis][1]), *run[axis + 1 :])
            else:
                merged[run] = key
                run = box
        merged[run] = key
    return merged


def _window_of(entries: list[_TargetDemand], target: int, box: Box) -> list[tuple[int, int, bool]]:
    """The window ``box`` lies in at ``target``: the hull of the windows of the entries there it overlaps."""
    windows = [entry.window for entry in entries if entry.target == target and meet(box, entry.box) is not None]
    return [
        (min(axis[0] for axis in axes), max(axis[1] for axis in axes), axes[0][2])
        for axes in zip(*windows, strict=True)
    ]


IndexSteps = tuple[tuple[int, int], ...]
"""Per axis of a tensor, the (step, first) of the indices read along it: ``first + step * k``."""


def _order_inputs_by_use(nodes: list[Node]) -> None:
    """Put a group's InEdges first, ordered by when the group first reads them.

    A group's InEdges accumulate from several passes, so their order would otherwise
    depend on which pass added them. The generated design takes its runtime arguments
    in this order, so it has to follow the group's own reads.
    """
    in_edges = [node for node in nodes if isinstance(node, InEdge)]
    rest = [node for node in nodes if not isinstance(node, InEdge)]
    read: list[Tensor] = []
    for node in rest:
        if isinstance(node, HasInputs):
            for tensor in node.inputs:
                if tensor not in read:
                    read.append(tensor)
    nodes[:] = sorted(in_edges, key=lambda e: read.index(e.outputs[0]) if e.outputs[0] in read else len(read))
    nodes.extend(rest)


class Workload(DiGraphWrapper[Node]):
    """A dataflow graph of nodes, immutable once built.

    Everything derived from the graph is computed once and kept, so a changed workload is a new
    ``Workload``: the graph is frozen and a mutating networkx call raises.
    """

    def __init__(self, nodes: Sequence[Node] = ()):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for node in nodes:
            if isinstance(node, HasInputs):
                for input in node.inputs:
                    # A state operand is read where it already sits, from one step of the node's
                    # own loop to the next. Nothing produces it and nothing carries it, so it has
                    # no edge -- an edge here would be the node waiting on itself.
                    if isinstance(node, HasIterationSpace) and is_state_operand(node, input):
                        continue
                    try:
                        pred = next(n for n in nodes if isinstance(n, HasOutputs) and input in n.outputs)
                    except StopIteration as e:
                        raise RuntimeError(f"Input tensor {input.name} for node {node.name} has no producer.") from e
                    graph.add_edge(pred, node)
        super().__init__(graph)
        nx.freeze(self)

    def dataflow_sort(self) -> tuple[Node, ...]:
        """Nodes in topological order, ties broken by the order they were added.

        The frontend adds nodes in the order the source graph lists them, so ties
        follow the dataflow rather than the node names. Renaming a tensor must not
        renumber the dimensions and solver variables derived from this order.
        """
        return self._dataflow_order

    @cached_property
    def _dataflow_order(self) -> tuple[Node, ...]:
        position = self.node_positions()
        return tuple(nx.lexicographical_topological_sort(self, key=position.__getitem__))

    def node_positions(self) -> dict[Node, int]:
        return {node: i for i, node in enumerate(self.nodes)}

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        node_names = ", ".join(getattr(n, "name", type(n).__name__) for n in nodes)
        return f"Workload(num_nodes={len(nodes)}, num_edges={len(edges)}, nodes=[{node_names}])"

    @cached_property
    def num_dims(self) -> int:
        return sum(node.num_dims for node in self.nodes if isinstance(node, HasIterationSpace))

    @cached_property
    def global_idxs(self) -> dict[Node, range]:
        """
        Determine unique global indices for each dimension in this workload
        """
        global_dimension_idxs: dict[Node, range] = {}
        idx = 0
        for node in self.dataflow_sort():
            if isinstance(node, HasIterationSpace):
                global_dimension_idxs[node] = range(idx, idx + node.num_dims)
                idx += node.num_dims
        return global_dimension_idxs

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        seen = set()
        tensors = []
        for node in self.get_iteration_space_nodes():
            for tensor in node.tensors:
                if tensor.name not in seen:
                    seen.add(tensor.name)
                    tensors.append(tensor)
        return tuple(tensors)

    def global_mapping(self, node: HasIterationSpace, mapping: AffineMap):
        return mapping.replace_dims_and_symbols(
            [AffineDimExpr(i) for i in self.global_idxs[node]], [], self.num_dims, 0
        )

    def _is_identity_relation(self, relation: AffineExpr) -> bool:
        """Whether a dimension relation ``d_a - d_b (+ const)`` merges two dims that are the same iteration axis."""
        row = AffineTransform.from_affine_map(AffineMap(self.num_dims, 0, (relation,))).A[0]
        return sorted(int(c) for c in row if c != 0) == [-1, 1]

    def _coupling(
        self, consumer: HasIterationSpace, produced: AffineExpr, read: AffineExpr, extent: int
    ) -> tuple[AffineExpr, int] | None:
        """The relation a producer->consumer axis of whole ``extent`` adds, and its stride: the identity, or for a read
        ``s*d + windows + c`` of a producer dim spanning ``s`` times ``d``, ``produced - s*d``, to which
        :attr:`_couplings` adds a remainder dim of size ``s``; none where the whole extents differ, as under an
        unpadded window."""
        if self._is_identity_relation(relation := produced - read):
            return relation, 1
        rows = AffineTransform.from_affine_map(AffineMap(self.num_dims, 0, (produced, read)))
        outputs = {
            self.global_idxs[consumer].start + d
            for t in consumer.outputs
            for d in map_dim_positions(consumer.get_mapping(t))
        }
        indexed = [int(d) for d in np.flatnonzero(rows.A[1]) if d in outputs]
        if sorted(rows.A[0][rows.A[0] != 0]) != [1] or rows.b[0] or len(indexed) != 1 or rows.A[1][indexed[0]] < 1:
            return None
        stride = int(rows.A[1][indexed[0]])
        local = AffineDimExpr(indexed[0] - self.global_idxs[consumer].start)
        spans = (
            n
            for t in consumer.outputs
            for r, n in zip(consumer.get_mapping(t).results, t.subview.source.type.get_shape(), strict=True)
            if r == local
        )
        if stride * next(spans, 0) != extent:
            return None
        return AffineDimExpr(int(np.flatnonzero(rows.A[0])[0])) - AffineDimExpr(indexed[0]) * stride, stride

    def dimension_relations(self) -> tuple[AffineExpr, ...]:
        return tuple(relation for relation, _ in self._couplings)

    @cached_property
    def _remainders(self) -> tuple[int, ...]:
        """The size of each remainder dim the couplings add as a column after the node dims."""
        return tuple(stride for _, stride in self._couplings if stride > 1)

    @cached_property
    def _couplings(self) -> tuple[tuple[AffineExpr, int], ...]:
        """Every dimension relation, and the stride of the windowed read it couples (1 for an identity), each strided
        read adding a remainder dim as a column after the node dims. An axis of a tensor several nodes read couples
        none of them where it would make two loops of one node one dim (:meth:`_held_axes`)."""
        candidates = self._candidate_couplings()
        held = self._held_axes(candidates)
        result: list[tuple[AffineExpr, int]] = []
        for relation, stride, shared in candidates:
            if shared in held:
                continue
            remainder = AffineDimExpr(self.num_dims + sum(s > 1 for _, s in result))
            result.append((relation - remainder if stride > 1 else relation, stride))
        return tuple(result)

    def _candidate_couplings(self) -> list[tuple[AffineExpr, int, _SharedAxis | None]]:
        """Each relation the dataflow suggests, its stride, and the axis of a tensor several nodes read it couples
        through, as its producer and axis (None for a tensor one node reads): a producer couples to each reader, and
        the readers of a workload input to each other."""
        candidates: list[tuple[AffineExpr, int, _SharedAxis | None]] = []
        for src, dst in self.edges:
            if isinstance(src, HasIterationSpace) and isinstance(dst, HasIterationSpace):
                try:
                    output = next(t for t in src.outputs if t in dst.inputs)
                except StopIteration as e:
                    raise RuntimeError(f"No shared tensor between nodes {src.name} and {dst.name}") from e
                mapping_out = self.global_mapping(src, src.get_mapping(output))
                mapping_in = self.global_mapping(dst, dst.get_mapping(output))
                full = output.subview.source.type.get_shape()
                shared = sum(isinstance(n, HasIterationSpace) for n in self.successors(src)) > 1
                axes = zip(mapping_out.results, mapping_in.results, full, strict=True)
                for axis, (expr_out, expr_in, extent) in enumerate(axes):
                    if (coupled := self._coupling(dst, expr_out, expr_in, extent)) is not None:
                        candidates.append((*coupled, (src, axis) if shared else None))
        for node in self.nodes:
            if isinstance(node, InEdge):
                assert len(node.outputs) == 1, "Only single output InEdge supported for now."
                output = node.outputs[0]
                all_users = [cast(HasIterationSpace, out) for (_, out) in self.out_edges(node)]
                for a, b in combinations(all_users, 2):
                    mapping_a = self.global_mapping(a, a.get_mapping(output))
                    mapping_b = self.global_mapping(b, b.get_mapping(output))
                    for axis, (expr_a, expr_b) in enumerate(zip(mapping_a.results, mapping_b.results, strict=True)):
                        relation = expr_a - expr_b
                        if self._is_identity_relation(relation):
                            candidates.append((relation, 1, (node, axis)))
        return candidates

    def _held_axes(self, candidates: list[tuple[AffineExpr, int, _SharedAxis | None]]) -> set[_SharedAxis]:
        """The axes of tensors several nodes read that couple none of their readers, because they play two roles: one
        coupling through them would make two loops of a node one dim, as the rows of attention's input reach its scores
        once as queries and once as keys. Such an axis is held whole, as the copy every reader slices. Tensors one node
        reads couple first, then the shared ones from the latest produced back, so the axis found is the one where the
        roles part rather than one a single role reaches further on."""
        position = {node: i for i, node in enumerate(self.dataflow_sort())}
        owner = {g: node for node, idxs in self.global_idxs.items() for g in idxs}
        links = sorted(
            ((self._joined_dims(relation), shared) for relation, _, shared in candidates),
            key=lambda link: (link[1] is not None, -position[link[1][0]] if link[1] else 0),
        )
        held: set[_SharedAxis] = set()
        while (conflict := self._role_conflict(links, held, owner)) is not None:
            held.add(conflict)
        return held

    def _joined_dims(self, relation: AffineExpr) -> list[int]:
        """The global dims ``relation`` joins."""
        row = AffineTransform.from_affine_map(AffineMap(self.num_dims, 0, (relation,))).A[0]
        return [int(g) for g in np.flatnonzero(row)]

    def _role_conflict(
        self, links: list[tuple[list[int], _SharedAxis | None]], held: set[_SharedAxis], owner: dict[int, Node]
    ) -> _SharedAxis | None:
        """The first shared axis in ``links`` (the global dims each coupling joins, and its shared axis) not ``held``
        whose coupling joins two dims of one node, the dims each other coupling joins being one."""
        parent = list(range(self.num_dims))
        nodes = [{owner[g]} for g in range(self.num_dims)]

        def find(g: int) -> int:
            while parent[g] != g:
                parent[g] = parent[parent[g]]
                g = parent[g]
            return g

        for dims, shared in links:
            if shared in held:
                continue
            for a, b in pairwise(dims):
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                if shared is not None and nodes[ra] & nodes[rb]:
                    return shared
                parent[rb] = ra
                nodes[ra] |= nodes[rb]
        return None

    def get_computation_nodes(self) -> tuple[ComputationNode, ...]:
        return tuple(cast(ComputationNode, node) for node in self.nodes if isinstance(node, ComputationNode))

    def get_transfer_nodes(self) -> tuple[TransferNode, ...]:
        return tuple(cast(TransferNode, node) for node in self.nodes if isinstance(node, TransferNode))

    def get_iteration_space_nodes(self) -> tuple[HasIterationSpace, ...]:
        return tuple(cast(HasIterationSpace, node) for node in self.nodes if isinstance(node, HasIterationSpace))

    def get_node_by_name(self, name: str) -> Node:
        for node in self.node_list:
            if node.name == name:
                return node
        raise KeyError(f"No node with name {name} found in workload.")

    def get_in_edges(self) -> tuple[InEdge, ...]:
        return tuple(cast(InEdge, node) for node in self.nodes if isinstance(node, InEdge))

    def get_out_edges(self) -> tuple[OutEdge, ...]:
        return tuple(cast(OutEdge, node) for node in self.nodes if isinstance(node, OutEdge))

    def split_fusion_groups(self, cut_points: list[str] | None = None) -> list["Workload"]:  # noqa: PLR0912
        """Split the workload at FusionEdge boundaries and explicit cut points.

        Each sub-workload is self-contained with InEdge at entries and OutEdge
        at exits. FusionEdge nodes are consumed: the FusionEdge's input tensor
        becomes an OutEdge in the preceding group, and its output tensor becomes
        an InEdge in the following group.

        When *cut_points* is provided, the listed node names act as additional
        group boundaries (the cut-point node stays in the preceding group).
        OutEdge/InEdge boundary pairs are created using the cut-point node's
        output tensor, analogous to FusionEdge boundaries.

        InEdge nodes (model inputs and initializers) are assigned to the group
        that contains their sole consumer. If an InEdge is consumed by nodes in
        multiple groups, it is duplicated into each consuming group.

        Args:
            cut_points: Optional list of node names at which to insert group
                boundaries.  When ``None`` (default), only FusionEdge
                boundaries are used (backward compatible).

        Returns:
            A list of Workloads. If there are no FusionEdge nodes and no cut
            points, returns ``[self]`` (single group).
        """
        from stream.workload.normalization import source_operator  # noqa: PLC0415 -- it imports this module

        fusion_edges = [n for n in self.nodes if isinstance(n, FusionEdge)]
        cut_point_set: set[str] = set(cut_points) if cut_points else set()
        if not fusion_edges and not cut_point_set:
            return [self]

        # Assign each non-FusionEdge, non-InEdge node to a group index.
        # Group boundaries are defined by FusionEdge nodes.
        topo_order = self.dataflow_sort()
        # A cut at a normalization expanded into sub-ops falls after the last of them
        last_sub_op = {source_operator(n): n.name for n in topo_order}
        cut_point_set = {last_sub_op.get(name, name) for name in cut_point_set}

        # Map each non-InEdge node to its group index
        node_to_group: dict[Node, int] = {}
        group_idx = 0
        for node in topo_order:
            if isinstance(node, FusionEdge):
                group_idx += 1
                # FusionEdge itself is not assigned to any group
                continue
            if isinstance(node, InEdge):
                # Defer InEdge assignment -- they go into the group(s) of their consumers
                continue
            node_to_group[node] = group_idx
            # Cut-point nodes end their group: subsequent nodes go into next group
            if cut_point_set and node.name in cut_point_set:
                group_idx += 1

        num_groups = group_idx + 1

        # Build node lists per group (excluding InEdges for now)
        group_nodes: list[list[Node]] = [[] for _ in range(num_groups)]
        for node in topo_order:
            if node in node_to_group:
                group_nodes[node_to_group[node]].append(node)

        # Assign InEdge nodes to the group(s) of their consumers. If consumed in multiple
        # groups, duplicate the InEdge into each. Their order is normalised below.
        for node in topo_order:
            if not isinstance(node, InEdge):
                continue
            consuming_groups = {node_to_group[c] for _, c in self.out_edges(node) if c in node_to_group}
            for grp in sorted(consuming_groups):
                group_nodes[grp].insert(0, node)

        # For each FusionEdge, add OutEdge to preceding group and InEdge to following group
        for fe in fusion_edges:
            # FusionEdge's input tensor -> OutEdge in preceding group
            assert len(fe.inputs) == 1, f"FusionEdge {fe.name} must have exactly 1 input"
            assert len(fe.outputs) == 1, f"FusionEdge {fe.name} must have exactly 1 output"

            # Find the group of the predecessor (producer of FusionEdge's input)
            preds = list(self.predecessors(fe))
            assert len(preds) == 1, f"FusionEdge {fe.name} must have exactly 1 predecessor"
            pred_group = node_to_group[preds[0]]

            # Find the group of the successor (consumer of FusionEdge's output)
            succs = list(self.successors(fe))
            assert len(succs) >= 1, f"FusionEdge {fe.name} must have at least 1 successor"
            succ_group = node_to_group[succs[0]]

            # Add OutEdge for the input tensor in the preceding group
            out_edge = OutEdge(
                name=f"{fe.name}_out",
                inputs=(fe.inputs[0],),
            )
            group_nodes[pred_group].append(out_edge)

            # Add InEdge for the output tensor in the following group
            in_edge = InEdge(
                name=f"{fe.name}_in",
                outputs=(fe.outputs[0],),
            )
            group_nodes[succ_group].insert(0, in_edge)

        self._add_cut_boundaries(node_to_group, group_nodes, cut_point_set)
        self._bridge_cross_group_edges(node_to_group, group_nodes)
        for nodes in group_nodes:
            _order_inputs_by_use(nodes)

        # Build sub-workloads
        sub_workloads = []
        for nodes in group_nodes:
            # A group of only boundary edges (no iteration space) would fail later in the affine solve; drop it here.
            if any(isinstance(node, HasIterationSpace) for node in nodes):
                sub_workloads.append(Workload(nodes))

        return sub_workloads

    @staticmethod
    def _add_cut_boundaries(
        node_to_group: dict[Node, int], group_nodes: list[list[Node]], cut_point_set: set[str]
    ) -> None:
        """Add the OutEdge/InEdge pair for each cut point whose output the next group reads.

        A cut point's output is not always consumed by the very next group: where the graph
        branches, it is read further downstream, and ``_bridge_cross_group_edges`` places
        that boundary in the group that does read it."""
        for node, grp in node_to_group.items():
            if not isinstance(node, ComputationNode) or node.name not in cut_point_set:
                continue
            assert len(node.outputs) == 1, f"Cut-point node {node.name} must have exactly 1 output"
            tensor = node.outputs[0]
            successor = group_nodes[grp + 1]
            if not any(isinstance(n, HasInputs) and tensor in n.inputs for n in successor):
                continue
            group_nodes[grp].append(OutEdge(name=f"{node.name}_cut_out", inputs=(tensor,)))
            successor.insert(0, InEdge(name=f"{node.name}_cut_in", outputs=(tensor,)))

    def _bridge_cross_group_edges(self, node_to_group: dict[Node, int], group_nodes: list[list[Node]]) -> None:
        """Add OutEdge/InEdge boundaries for any data edge crossing a group boundary without passing
        through a FusionEdge or cut-point (e.g. attention's V, which skips the softmax barrier to feed
        the context epilogue). Without this the consumer group would have an input with no producer."""

        def has_inedge(nodes: list[Node], tensor: Tensor) -> bool:
            return any(isinstance(n, InEdge) and tensor in n.outputs for n in nodes)

        def has_outedge(nodes: list[Node], tensor: Tensor) -> bool:
            return any(isinstance(n, OutEdge) and tensor in n.inputs for n in nodes)

        for producer, group in node_to_group.items():
            for consumer in self.successors(producer):
                if consumer not in node_to_group or node_to_group[consumer] == group:
                    continue
                consumer_group = node_to_group[consumer]
                for tensor in set(producer.outputs) & set(consumer.inputs):  # type: ignore[attr-defined]
                    if not has_outedge(group_nodes[group], tensor):
                        group_nodes[group].append(OutEdge(name=f"{tensor.name}_bridge_out", inputs=(tensor,)))
                    if not has_inedge(group_nodes[consumer_group], tensor):
                        bridge_in = InEdge(name=f"{tensor.name}_bridge_in", outputs=(tensor,))
                        group_nodes[consumer_group].insert(0, bridge_in)

    def get_dimension_sizes(self) -> tuple[int, ...]:
        """The extent of every global dimension slot, then of every remainder dim."""
        return self._dimension_sizes

    @cached_property
    def _dimension_sizes(self) -> tuple[int, ...]:
        result_to_shape: list[tuple[AffineExpr, int]] = []
        for node in self.get_iteration_space_nodes():
            for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
                global_mapping = self.global_mapping(node, mapping)
                for expr, sz in zip(global_mapping.results, _span(tensor), strict=True):
                    result_to_shape.append((expr, sz))

        # Step 1: direct read for dims that appear as pure AffineDimExpr
        dim_to_size: dict[int, int] = {}
        for expr, sz in result_to_shape:
            if isinstance(expr, AffineDimExpr) and expr.position not in dim_to_size:
                dim_to_size[expr.position] = sz

        for node in self.get_computation_nodes():
            for d, size in node.window_extents:
                dim_to_size.setdefault(self.global_idxs[node].start + d, size)

        coefficients = [(expr, sz, affine_coefficients(expr, self.num_dims)[1]) for expr, sz in result_to_shape]
        missing = sorted(set(range(self.num_dims)) - set(dim_to_size))
        while found := next(
            (
                (d, expr, sz, c)
                for d in missing
                for expr, sz, c in coefficients
                if c[d] and not any(c[other] for other in missing if other != d)
            ),
            None,
        ):
            d, expr, sz, c = found
            high = affine_bounds(expr, [dim_to_size.get(k, 1) for k in range(self.num_dims)])[1]
            dim_to_size[d] = (sz - 1 - high) // c[d] + 1
            missing.remove(d)

        assert len(dim_to_size) == self.num_dims, (
            f"Could not determine sizes for all {self.num_dims} dims: "
            f"missing {sorted(set(range(self.num_dims)) - set(dim_to_size.keys()))}"
        )
        return (*(dim_to_size[i] for i in range(self.num_dims)), *self._remainders)

    def get_dims(self, node: HasIterationSpace) -> list[LayerDim]:
        global_idxs = self.global_idxs
        _, expressions = self.unique_dimensions()
        start_idx = global_idxs[node].start
        stop_idx = global_idxs[node].stop
        dims = expressions[start_idx:stop_idx]
        return dims

    def get_dimension_size(self, dim: LayerDim) -> int:
        """The extent of ``dim``, a unique dim or a node dim: a unique dim spans its most downstream node's extent along
        it, which is where its column in the solve stands (a windowed reader's window spans more)."""
        return self._sizes_by_dim[dim]

    @cached_property
    def _sizes_by_dim(self) -> dict[AffineExpr, int]:
        """The extent of each dim expression at its last occurrence, as :meth:`get_dimension_size` reads it."""
        return dict(zip(self.unique_dimensions()[1], self.get_dimension_sizes(), strict=True))

    def unique_dimensions(self) -> tuple[tuple[LayerDim, ...], tuple[AffineExpr, ...]]:
        """The workload's independent dimensions, and each global dimension as an expression of them."""
        return self._unique_dimensions

    @cached_property
    def _unique_dimensions(self) -> tuple[tuple[LayerDim, ...], tuple[AffineExpr, ...]]:
        relations = AffineMap(self.num_dims + len(self._remainders), 0, self.dimension_relations())
        transform = AffineTransform.from_affine_map(relations)

        A_sp = sp.Matrix(transform.A)
        b_sp = sp.Matrix(transform.b).reshape(len(transform.b), 1)
        n_vars = transform.A.shape[1]

        # Solve A*x = -b via augmented matrix RREF: [A | -b]
        augmented = A_sp.row_join(-b_sp)
        rref_aug, pivots = augmented.rref()

        free_vars = [i for i in range(n_vars) if i not in pivots]

        # Null space basis: homogeneous solutions (from free variable columns of A)
        basis_vectors = []
        for free in free_vars:
            v = sp.zeros(n_vars, 1)
            v[free] = 1
            for row, pivot in enumerate(pivots):
                v[pivot] = -rref_aug[row, free]
            basis_vectors.append(v)

        N = sp.Matrix.hstack(*basis_vectors)
        z_syms = sp.symbols(f"z0:{len(free_vars)}")

        # Particular solution from the last column of the augmented RREF
        x_p = sp.zeros(n_vars, 1)
        for row, pivot in enumerate(pivots):
            x_p[pivot] = rref_aug[row, n_vars]

        x = N * sp.Matrix(z_syms) + x_p

        dim_values = tuple(sympy_to_xdsl(sp.simplify(expr)) for expr in x)
        z = tuple(LayerDim(position=i, prefix="z") for i in range(len(free_vars)))
        return z, dim_values

    def get_unique_dims_inter_core_tiling(self, node: ComputationNode, mapping: "Mapping") -> InterCoreTiling:
        """Convert inter_core_tiling dimensions from LayerDim to unique workload indices."""
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        unique_node_dims = self.get_dims(node)
        if not node_mapping.inter_core_tiling:
            return ()
        converted_tiling: list[tuple[LayerDim, int]] = []
        all_tilings_equal = all(t == node_mapping.inter_core_tiling[0] for t in node_mapping.inter_core_tiling)
        assert all_tilings_equal, f"Multiple different inter-core tilings for node {node.name} not supported for now."
        for dim, factor in node_mapping.inter_core_tiling[0]:
            unique_dim = dim if "z" in str(dim) else self.leading_dim(unique_node_dims[dim.position])[0]
            converted_tiling.append((unique_dim, factor))
        return tuple(converted_tiling)

    @cached_property
    def _leading_dims(self) -> dict[AffineExpr, tuple[LayerDim, int]]:
        """:meth:`leading_dim` of each dim expression."""
        unique_dims, expressions = self.unique_dimensions()
        leading: dict[AffineExpr, tuple[LayerDim, int]] = {}
        for dim in expressions:
            coefficients = affine_coefficients(dim, len(unique_dims))[1]
            k = max(range(len(unique_dims)), key=lambda k: abs(coefficients[k]))
            leading[dim] = unique_dims[k], coefficients[k]
        return leading

    def leading_dim(self, dim: AffineExpr) -> tuple[LayerDim, int]:
        """The unique dim a node dim steps along fastest, and its coefficient: ``z`` and 2 for a strided reader's
        producer dim ``2*z + r``, so splitting or tiling the node dim splits or tiles ``z``."""
        return self._leading_dims[dim]

    def get_tensor_shape_with_dimension_sizes(
        self,
        tensor: Tensor,
        dimension_sizes: dict[LayerDim, int],
        accessor: HasIterationSpace | None = None,
        at: dict[LayerDim, int] | None = None,
        readers: bool = False,
    ) -> tuple[int, ...]:
        """The extent per axis of what ``accessor`` touches of ``tensor`` (by default its producer, or its ``readers``,
        else its other users) when each unique dim spans ``dimension_sizes``: an interior tile, its window at most the
        whole tensor, but clipped to the tensor along the axes of the dims ``at`` places at a tile index."""
        nodes = self._accessors(tensor, accessor, readers)
        return self._extents(tensor, nodes, dimension_sizes, at or {}, self._tile_index_steps(tensor, nodes))

    def _accessors(self, tensor: Tensor, accessor: HasIterationSpace | None, readers: bool) -> list[HasIterationSpace]:
        """``accessor``, else the producers of ``tensor`` (or its ``readers``), else its other users."""
        users = [n for n in self.get_iteration_space_nodes() if tensor in n.tensors]
        chosen = [n for n in users if tensor in (n.inputs if readers else n.outputs)]
        return [accessor] if accessor else chosen or users

    def _extents(
        self,
        tensor: Tensor,
        nodes: Sequence[HasIterationSpace],
        dimension_sizes: dict[LayerDim, int],
        at: dict,
        index_steps: IndexSteps,
    ) -> tuple[int, ...]:
        """The indices per axis ``nodes`` touch of ``tensor``, counted on ``index_steps``."""
        return tuple(
            max(0, high - low + 1)
            for low, high, _ in self._bounds(tensor, nodes, dimension_sizes, at, index_steps=index_steps)
        )

    def _tile_index_steps(
        self, tensor: Tensor, nodes: Sequence[HasIterationSpace], index_steps: IndexSteps | None = None
    ) -> IndexSteps:
        """The index steps a tile ``nodes`` touch is held on: every index where one of them writes ``tensor``, else
        ``index_steps`` (by default the readers')."""
        if any(tensor in node.outputs for node in nodes):
            return _every_index(tensor)
        return index_steps or self.index_steps(tensor)

    def index_steps(self, tensor: Tensor) -> IndexSteps:
        """The index steps the computation nodes reading the buffer of ``tensor`` read it on."""
        return self._index_steps_by_buffer.get(tensor.subview.source) or _every_index(tensor)

    @cached_property
    def _index_steps_by_buffer(self) -> dict[Any, IndexSteps]:
        return self._index_steps_of({z: self.get_dimension_size(z) for z in self.unique_dimensions()[0]})

    def _index_steps_of(self, sizes: dict[LayerDim, int]) -> dict[Any, IndexSteps]:
        """Per buffer a node reads, the index steps its computation nodes read it on, each unique dim spanning
        ``sizes``."""
        users: dict[Any, list[tuple[HasInputs, Tensor]]] = {}
        for node in self.nodes:
            if isinstance(node, HasInputs):
                for tensor in node.inputs:
                    users.setdefault(tensor.subview.source, []).append((node, tensor))
        return {source: self._index_steps_for(found, sizes) for source, found in users.items()}

    def _index_steps_for(self, users: list[tuple[HasInputs, Tensor]], sizes: dict[LayerDim, int]) -> IndexSteps:
        """The index steps ``users`` of one buffer read it on: per axis the gcd of the coefficients of the dims that
        move the index and of the differences between the readers' first indices; every index if an edge uses it."""
        reads = [(n, t) for n, t in users if isinstance(n, ComputationNode)]
        if not reads or any(not isinstance(n, ComputationNode | TransferNode) for n, _ in users):
            return _every_index(users[0][1])
        unique_dims, dim_values = self.unique_dimensions()
        index_steps = []
        for axis in range(len(users[0][1].subview.source.type.get_shape())):
            step, firsts = 0, []
            for node, t in reads:
                index = self.global_mapping(node, node.get_mapping(t)).results[axis]
                first, coefficients = affine_coefficients(
                    index.replace_dims_and_symbols(dim_values, ()), len(unique_dims)
                )
                moving = (abs(c) for c, z in zip(coefficients, unique_dims, strict=True) if c and sizes[z] > 1)
                step = gcd(step, *moving)
                firsts.append(first)
            step = gcd(step, *(first - firsts[0] for first in firsts)) or 1
            index_steps.append((step, firsts[0] % step))
        return tuple(index_steps)

    def strided_view(self, tensor: Tensor) -> Tensor:
        """The indices of the tile ``tensor`` its readers read, as a strided view of the same range."""
        index_steps = self.index_steps(tensor)
        offsets, strides = tensor.subview.static_offsets.get_values(), tensor.subview.static_strides.get_values()
        if all(stride == step for stride, (step, _) in zip(strides, index_steps, strict=True)):
            return tensor
        ranges = [
            _stepped_range(offset, offset + (n - 1) * stride, *on)
            for offset, n, stride, on in zip(offsets, tensor.shape, strides, index_steps, strict=True)
        ]
        return self._tile(tensor, tuple(max(0, high - low + 1) for low, high in ranges), index_steps)

    def _bounds(
        self,
        tensor: Tensor,
        nodes: Sequence[HasIterationSpace],
        dimension_sizes: dict[LayerDim, int],
        at: dict,
        origin: dict[LayerDim, int] | None = None,
        *,
        index_steps: IndexSteps | None = None,
    ) -> list[tuple[int, int, bool]]:
        """The inclusive index range per axis ``nodes`` touch of ``tensor`` as for the extents above, and whether the
        axis is placed; a dim of ``origin`` starts at the index it gives, in tensor coordinates but not placed. The
        range counts the indices on ``index_steps`` (by default the readers')."""
        index_steps = index_steps or self.index_steps(tensor)
        unique_dims, dim_values = self.unique_dimensions()
        sizes = [dimension_sizes[z] for z in unique_dims]
        origin = origin or {}
        offset = [origin.get(z, at.get(z, 0) * n) for z, n in zip(unique_dims, sizes, strict=True)]
        bounds = []
        for axis, size in enumerate(tensor.subview.source.type.get_shape()):
            ranges, placed = [], False
            for node in nodes:
                index = self.global_mapping(node, node.get_mapping(tensor)).results[axis]
                index = index.replace_dims_and_symbols(dim_values, ())
                low, high = affine_bounds(index, sizes)
                coefficients = affine_coefficients(index, len(sizes))[1] if at or origin else [0] * len(sizes)
                if any(c and (z in at or z in origin) for c, z in zip(coefficients, unique_dims, strict=True)):
                    placed = placed or any(c and z in at for c, z in zip(coefficients, unique_dims, strict=True))
                    shift = sum(c * o for c, o in zip(coefficients, offset, strict=True))
                    ranges.append((max(low + shift, 0), min(high + shift, size - 1)))
                else:
                    # A window spanning the axis is all of it; a narrower one keeps an interior tile's extent.
                    ranges.append((0, size - 1) if high - low + 1 >= size else (low, high))
                ranges[-1] = _stepped_range(*ranges[-1], *index_steps[axis])
            bounds.append((min(r[0] for r in ranges), max(r[1] for r in ranges), placed))
        return bounds

    def _tile_sizes(self, tiling: InterCoreTiling) -> dict[LayerDim, int]:
        """The extent of every unique dim on one core of ``tiling``."""
        factors = dict(tiling)
        return {z: self.get_dimension_size(z) // factors.get(z, 1) for z in self.unique_dimensions()[0]}

    @staticmethod
    def _position(tiling: InterCoreTiling, core: int | None) -> dict[LayerDim, int]:
        """The tile index along each dim of ``tiling`` of the core at position ``core``, the last dim fastest."""
        if core is None or not tiling:
            return {}
        return {dim: int(i) for (dim, _), i in zip(tiling, np.unravel_index(core, [f for _, f in tiling]), strict=True)}

    def get_tensor_shape_with_tiling(
        self,
        tensor: Tensor,
        tiling: InterCoreTiling,
        accessor: HasIterationSpace | None = None,
        core: int | None = None,
        readers: bool = False,
    ) -> tuple[int, ...]:
        """The tile of ``tensor`` on one core of ``tiling``: an interior one, or the one at position ``core``."""
        return self.get_tensor_shape_with_dimension_sizes(
            tensor, self._tile_sizes(tiling), accessor, self._position(tiling, core), readers
        )

    def _reader(self, tensor: Tensor, node: Node) -> tuple[Tensor, HasIterationSpace] | None:
        """For a transfer's copy ``tensor``, the computation node the copy reaches through any further transfers and
        the copy of the tensor that node reads, whose window the copy holds; None for anything else."""
        if not isinstance(node, TransferNode) or tensor not in node.outputs:
            return None
        succ = next(n for n in self.successors(node) if isinstance(n, HasInputs) and tensor in n.inputs)
        if isinstance(succ, TransferNode) and len(succ.outputs) == 1:
            return self._reader(succ.outputs[0], succ)
        return (tensor, succ) if isinstance(succ, ComputationNode) else None

    def _tiling(self, node: Node, mapping: "Mapping") -> InterCoreTiling:
        tilings = mapping.get(node).inter_core_tiling
        assert all(t == tilings[0] for t in tilings), "Multiple different tilings not implemented yet."
        return tilings[0] if tilings else ()

    @staticmethod
    def _tile(tensor: Tensor, shape: tuple[int, ...], index_steps: IndexSteps | None = None) -> Tensor:
        """A tile of ``shape`` of ``tensor``'s buffer, its indices on ``index_steps``."""
        index_steps = index_steps or _every_index(tensor)
        subview = SubviewOp.from_static_parameters(
            source=tensor.subview.source,
            source_type=tensor.subview.source.type,
            offsets=[first for _, first in index_steps],
            sizes=shape,
            strides=[step for step, _ in index_steps],
        )
        return Tensor(name=tensor.name, operand_type=tensor.operand_type, shape=shape, subview=subview)

    def get_tensor_single_core(
        self, tensor: Tensor, node: HasOutputs, mapping: "Mapping", core: int | None = None
    ) -> Tensor:
        """The tile of ``tensor`` ``node`` holds on one core, interior or the one at position ``core``; a transfer's
        copy holds the window of the computation node it reaches, under that node's split where it feeds the node
        directly, under the transfer's own where it is staged for another transfer."""
        read, accessor = self._reader(tensor, node) or (tensor, cast(HasIterationSpace, node))
        shape = self.get_tensor_shape_with_tiling(read, self._window_tiling(node, accessor, mapping), accessor, core)
        return tensor if shape == tensor.shape else self._tile(tensor, shape, self._tile_index_steps(read, [accessor]))

    def _window_tiling(self, node: Node, accessor: HasIterationSpace, mapping: "Mapping") -> InterCoreTiling:
        """The split of ``accessor`` where it reads its tile from ``node`` directly, else ``node``'s own."""
        if isinstance(accessor, ComputationNode) and accessor in self.successors(node):
            return self.get_unique_dims_inter_core_tiling(accessor, mapping)
        return self._tiling(node, mapping)

    def get_windows(
        self,
        tensor: Tensor,
        transfer: TransferNode,
        mapping: "Mapping",
        dims: Sequence[LayerDim],
        *,
        temporal: bool = False,
    ) -> dict[LayerDim, tuple[int, int]]:
        """Along each of ``dims`` that indexes the output of the node a transfer's copy reaches and slides its read with
        an overlap, the tensor axis it slides and the halo a tile shares with the next: the window less its step. A
        ``temporal`` loop steps one core by the tile times the split of its dim."""
        if (found := self._reader(tensor, transfer)) is None:
            return {}
        (read, reader), (unique_dims, dim_values) = found, self.unique_dimensions()
        tiling = self._tiling(transfer, mapping)
        sizes, factors = self._tile_sizes(tiling), dict(tiling) if temporal else {}
        shape = self.get_tensor_shape_with_dimension_sizes(read, sizes, reader)
        index_steps = self._tile_index_steps(read, [reader])
        sliding = set(dims) & set(self.get_tensor_dimensions(reader.outputs[0]))
        windows: dict[LayerDim, tuple[int, int]] = {}
        for axis, expr in enumerate(self.global_mapping(reader, reader.get_mapping(read)).results):
            index = expr.replace_dims_and_symbols(dim_values, ())
            for z, c in zip(unique_dims, affine_coefficients(index, len(unique_dims))[1], strict=True):
                step = abs(c) * sizes[z] * factors.get(z, 1) // index_steps[axis][0]
                if z in sliding and c and (halo := shape[axis] - step) > 0:
                    windows.setdefault(z, (axis, halo))
        return windows

    def get_tensor_of_transfer_to_single_core(
        self,
        tensor: Tensor,
        transfer: TransferNode,
        mapping: "Mapping",
        core: int | None = None,
        ssis: SteadyStateIterationSpace | None = None,
    ) -> Tensor:
        """What ``transfer`` moves of ``tensor`` to one core per firing, interior or at position ``core``: the window
        of the node it reaches, less the halo each sliding loop of ``ssis`` keeps resident."""
        succ = next(n for n in self.successors(transfer) if tensor in n.inputs)
        if isinstance(succ, OutEdge):
            tiling = tuple()
        elif isinstance(succ, TransferNode):
            tiling = self.get_unique_dims_inter_core_tiling(transfer, mapping)
        elif isinstance(succ, ComputationNode):
            tiling = self.get_unique_dims_inter_core_tiling(succ, mapping)
        else:
            raise TypeError(f"Unexpected successor type {type(succ)} for transfer node {transfer.name}")
        read, reader = self._reader(tensor, transfer) or (tensor, None)
        shape = list(self.get_tensor_shape_with_tiling(read, tiling, reader, core))
        for axis, halo in self.sliding_halos(tensor, transfer, mapping, ssis).values():
            shape[axis] -= halo
        return self._tile(tensor, tuple(shape), self._tile_index_steps(read, self._accessors(read, reader, False)))

    def sliding_halos(
        self, tensor: Tensor, transfer: TransferNode, mapping: "Mapping", ssis: SteadyStateIterationSpace | None
    ) -> dict[LayerDim, tuple[int, int]]:
        """Per dim of a loop of ``ssis`` that slides the window of a transfer's copy, the axis it slides and the halo
        it keeps resident there."""
        sliding = [v.dimension for v in _sliding_loops(ssis.get_temporal_variables() if ssis else [])]
        return self.get_windows(tensor, transfer, mapping, sliding, temporal=True) if sliding else {}

    def kept_halos(
        self, tensor: Tensor, transfer: TransferNode, mapping: "Mapping", ssis: SteadyStateIterationSpace | None
    ) -> dict[int, int]:
        """Per reuse stop of a transfer's copy, the elements of halo it holds on one core in addition to its resident
        windows: each sliding loop outside the stop keeps its halo across the trips of the loops inside it, less what
        the resident windows already hold."""
        if ssis is None or (found := self._reader(tensor, transfer)) is None:
            return {}
        read, reader = found
        halos = self.sliding_halos(tensor, transfer, mapping, ssis)
        sizes = self._tile_sizes(self._window_tiling(transfer, reader, mapping))
        loops = ssis.get_applicable_temporal_variables()

        def slab(loop: IterationVariable, inner: list[IterationVariable]) -> int:
            swept = dict(sizes)
            for v in inner:
                if v.relevant and v.dimension != loop.dimension:
                    swept[v.dimension] *= v.size
            shape = self.get_tensor_shape_with_dimension_sizes(read, swept, reader)
            axis, halo = halos[loop.dimension]
            return prod(shape) // shape[axis] * min(halo, shape[axis])

        kept: dict[int, int] = {}
        for i, loop in enumerate(loops):
            if loop.dimension not in halos or not _sliding_loops([loop]):
                continue
            for stop in range(-1, i):
                if extra := slab(loop, loops[:i]) - slab(loop, loops[: stop + 1]):
                    kept[stop] = kept.get(stop, 0) + extra
        return kept

    def get_transfer_overlaps(
        self, transfer: TransferNode, mapping: "Mapping", ssis: SteadyStateIterationSpace | None = None
    ) -> dict[tuple[int, int], float]:
        """Elements per firing the source at position i serves the target at position j, where the windows of the nodes
        the copies reach overlap neighbouring sources' tiles, averaged over the target's trips; empty where none does,
        the positions then pairing up."""
        boxes, trips = self._window_boxes(transfer, mapping, self._tiling(transfer, mapping), ssis)
        overlaps = {pair: sum(_union(found) for found in per_trip.values()) / trips for pair, per_trip in boxes.items()}
        return {pair: n for pair, n in overlaps.items() if n}

    def _window_boxes(
        self,
        transfer: TransferNode,
        mapping: "Mapping",
        tiling: InterCoreTiling,
        ssis: SteadyStateIterationSpace | None = None,
    ) -> tuple[dict[tuple[int, int], dict[int, list[list[tuple[int, int]]]]], int]:
        """Per trip of the target (:meth:`_overlapping_split_loops`), the boxes of its tensor the source at position i
        serves the target at position j, for the copies whose reader slides an overlapping window over them, and the
        number of trips."""
        source, src = transfer.inputs[0], next(self.predecessors(transfer))
        src_tiling = self._tiling(src, mapping) if isinstance(src, ComputationNode) else ()
        placed = {d for d, _ in (*tiling, *src_tiling)}
        fused = {v.dimension: v.size for v in ssis.get_temporal_variables()} if ssis else {}
        trips = self._overlapping_split_loops(transfer, mapping, placed, fused)
        target_trips = [dict(zip(trips, steps, strict=True)) for steps in product(*(range(n) for n in trips.values()))]
        sources, factors = prod(f for _, f in src_tiling), dict(tiling)

        def bounds(t: Tensor, node: Node, core: int, steps: dict[LayerDim, int]) -> list[tuple[int, int, bool]]:
            if not isinstance(node, HasIterationSpace):
                return [(0, n - 1, True) for n in t.shape]
            at = dict.fromkeys(placed, 0) | self._position(tiling, core)
            for dim, step in steps.items():
                at[dim] = at.get(dim, 0) + step * factors.get(dim, 1)
            return self._bounds(t, [node], self._tile_sizes(tiling), at)

        boxes: dict[tuple[int, int], dict[int, list[list[tuple[int, int]]]]] = {}
        for tensor in transfer.outputs:
            found = self._reader(tensor, transfer)
            if found is None or not self.get_windows(tensor, transfer, mapping, [d for d, _ in tiling]):
                continue
            moved = self.get_tensor_of_transfer_to_single_core(tensor, transfer, mapping, ssis=ssis)
            for i in range(sources):
                tiles = self._trip_tiles([(source, src, src_tiling, i, sources)], placed, trips)
                for j, (k, steps) in product(range(prod(f for _, f in tiling)), enumerate(target_trips)):
                    want = bounds(found[0], found[1], j, steps)
                    for have in tiles:
                        box = [
                            (max(h[0], w[0]), min(h[1], w[1])) if w[2] else (0, m - 1)
                            for h, w, m in zip(have, want, moved.shape, strict=True)
                        ]
                        if all(lo <= hi for lo, hi in box):
                            boxes.setdefault((i, j), {}).setdefault(k, []).append(box)
        return boxes, len(target_trips)

    def _walked_blocks(
        self, placement: "EdgePlacement | None", tensor: Tensor
    ) -> list[list[list[tuple[int, int]]]] | None:
        """Per iteration walking the split axis, the box each position of ``placement`` holds in iteration-0
        coordinates; None unless it splits ``tensor``."""
        if placement is None or placement.split is None:
            return None
        shape, axis, index_steps = tensor.subview.source.type.get_shape(), placement.split, self.index_steps(tensor)
        whole = [_stepped_range(0, n - 1, *on) for n, on in zip(shape, index_steps, strict=True)]
        extent = whole[axis][1] + 1
        step = min(tensor.shape[axis], extent)
        blocks = [
            _stepped_range(*placement.block(s, shape[axis]), *index_steps[axis]) for s in range(len(placement.memories))
        ]
        return [
            [[*whole[:axis], (lo - k * step, hi - k * step), *whole[axis + 1 :]] for lo, hi in blocks]
            for k in range(ceil(extent / step))
        ]

    def placement_shares(
        self, placement: "EdgePlacement | None", tensor: Tensor, boxes: Sequence
    ) -> dict[Memory, float] | None:
        """The share of ``boxes`` each memory of ``placement`` serves over the walk of its split axis; None unless it
        splits ``tensor``."""
        if placement is None or not (walk := self._walked_blocks(placement, tensor)):
            return None
        covered = [
            sum(box_bits([part for box in boxes if (part := meet(box, blocks[s]))], 1) for blocks in walk)
            for s in range(len(walk[0]))
        ]
        if not (total := sum(covered)):
            return None
        shares: dict[Memory, float] = {}
        for memory, bits in zip(placement.memories, covered, strict=True):
            shares[memory] = shares.get(memory, 0.0) + bits / total
        return shares

    def get_transfer_demand(  # noqa: PLR0913
        self,
        transfer: TransferNode,
        mapping: "Mapping",
        tiling: InterCoreTiling,
        n_sources: int,
        n_targets: int,
        by_spatial_index: bool,
        targets: Sequence[Memory],
        reads_in: Callable[[ComputationNode, Tensor], tuple[Memory, ...]],
        cost: Callable[[Sequence[DemandItem]], tuple[float, ...]],
        *,
        ssis: Callable[[Node | Tensor], SteadyStateIterationSpace | None],
        fusion_splits: dict[LayerDim, int],
        allow_local_copy: bool,
        shares_memory: Callable[[int, int], bool],
    ) -> tuple[DemandItem, ...]:
        """What each of the ``n_targets`` targets of ``transfer`` reads per firing from which of the ``n_sources``:
        each copy's window less what its sliding loop keeps, a box several copies read once. With ``allow_local_copy``,
        a copy may be a local copy from a larger copy in the same memory. A walked split placement is taken at its
        busiest (``cost``). ``by_spatial_index`` pairs target ``j`` with sources ``j``, ``j + m``, ..., as code
        generation does."""
        source, src = transfer.inputs[0], next(self.predecessors(transfer))
        src_tiling = self._tiling(src, mapping) if isinstance(src, ComputationNode | TransferNode) else ()
        placed = {d for d, _ in (*tiling, *src_tiling)}
        held = (
            [(source, src)]
            if isinstance(src, ComputationNode)
            else [found for t in transfer.outputs if (found := self._reader(t, transfer))] or [(source, src)]
        )
        trips = self._overlapping_split_loops(transfer, mapping, placed, fusion_splits)
        haves = [
            self._trip_tiles([(*read, src_tiling, s, n_sources) for read in held], placed, trips)
            for s in range(n_sources)
        ]
        partial = isinstance(src, ComputationNode) and self.splits_reduction(src, src_tiling)

        def contained(smaller: _Read, larger: _Read) -> bool:
            return self._always_within(smaller, larger, fusion_splits, placed)

        out = next((n for n in self.successors(transfer) if isinstance(n, OutEdge)), None)
        held_at, target = mapping.placement_of(src), mapping.placement_of(out) if out else None
        held_walk, target_walk = self._walked_blocks(held_at, source), self._walked_blocks(target, source)
        nest = (
            None
            if by_spatial_index or held_walk or target_walk
            else self._firing_nest(transfer, placed, ssis, fusion_splits, set(trips))
        )
        wanted = [
            self._target_demand(
                transfer,
                mapping,
                tiling,
                placed,
                targets,
                None if by_spatial_index else reads_in,
                ssis,
                contained if allow_local_copy else None,
                firing=firing,
            )
            for firing in ([None] if nest is None else [(nest, s) for s in range(len(nest.trip_indices))])
        ]
        dims = [d for d, _ in tiling]
        spread = {c for c, copy in enumerate(transfer.outputs) if not self.get_windows(copy, transfer, mapping, dims)}

        def firing(entries: list[_TargetDemand], held_blocks: list | None) -> list[DemandItem]:
            items: list[DemandItem] = []

            def holders(t: int, box: Box, want: list[tuple[int, int, bool]]) -> dict[Box, list[int]]:
                found: dict[Box, list[int]] = {}
                tiles = [[tuple(map(tuple, b))] for b in held_blocks] if held_blocks else haves
                for s, boxes in enumerate(tiles):
                    for have in boxes:
                        held = have if held_blocks else _source_part_of_window(have, want)
                        if (part := meet(box, held)) is not None:
                            found.setdefault(part, []).append(s)
                return found

            for t, box, want, copies, local_copy, copied_from in (
                (e.target, e.box, e.window, e.copies, e.local_copy, e.copied_from) for e in entries
            ):
                if local_copy:
                    here = {} if partial else holders(t, box, want)
                    in_place = {
                        part: s for part, found in here.items() if (s := [s for s in found if shares_memory(s, t)])
                    }
                    items += [DemandItem(t, tuple(s), part, False, copies) for part, s in in_place.items()]
                    items += [DemandItem(t, (), rest, False, copies, copied_from) for rest in less(box, in_place)]
                elif by_spatial_index and spread.issuperset(copies):
                    narrow = min(n_sources, n_targets)
                    feeding = [s for s in range(n_sources) if s % narrow == t % narrow]
                    parts = zip(feeding, _split(list(box), len(feeding)), strict=True)
                    items += [
                        DemandItem(t, (s,), part, False, copies)
                        for s, part in parts
                        if all(lo <= hi for lo, hi in part)
                    ]
                else:
                    items += [
                        DemandItem(t, tuple(sources), part, partial, copies)
                        for part, sources in holders(t, box, want).items()
                    ]
            return items

        def demand(held_blocks: list | None, target_blocks: list | None) -> list[DemandItem]:
            if target_blocks:
                entries = [
                    _TargetDemand(t, tuple(block), [(lo, hi, True) for lo, hi in block], (0,), False)
                    for t, block in enumerate(target_blocks)
                ]
                return firing(entries, held_blocks)
            return _over_firings([firing(entries, held_blocks) for entries in wanted])

        if held_walk and target_walk and held_at and target and held_at.split == target.split:
            walks = zip(held_walk, target_walk, strict=True)
        else:
            walks = product(held_walk or [None], target_walk or [None])
        return tuple(_busiest([demand(held, target) for held, target in walks], cost))

    def neighbour_wait(self, transfer: TransferNode, mapping: "Mapping", fusion_splits: dict[LayerDim, int]) -> int:
        """The most firings of the fused nest a reader of ``transfer`` reads a tile before its producer's own firing
        makes it, along the fused loops over a dim the producer's split places: the reader waits that long."""
        src = next(iter(self.predecessors(transfer)))
        if not isinstance(src, ComputationNode):
            return 0
        src_tiling = self._tiling(src, mapping)
        placed = {d for d, _ in (*self._tiling(transfer, mapping), *src_tiling)}
        split = {d for d, f in src_tiling if f > 1}
        crossed = self._overlapping_split_loops(transfer, mapping, placed, fusion_splits)
        if not (trips := {d: n for d, n in crossed.items() if d in split}):
            return 0
        strides = {d: prod(list(fusion_splits.values())[:k]) for k, d in enumerate(fusion_splits)}
        steps = [dict(zip(trips, at, strict=True)) for at in product(*(range(n) for n in trips.values()))]

        def firing(at: dict[LayerDim, int]) -> int:
            return sum(k * strides[d] for d, k in at.items())

        sources = prod(f for _, f in src_tiling)
        made = [
            (firing(at), self._read_box((transfer.inputs[0], src, src_tiling, i, sources), placed, at))
            for i, at in product(range(sources), steps)
        ]
        lead = 0
        for read, reader in (found for copy in transfer.outputs if (found := self._reader(copy, transfer))):
            tiling = self._tiling(reader, mapping)
            cores = prod(f for _, f in tiling)
            for p, at in product(range(cores), steps):
                window = self._read_box((read, reader, tiling, p, cores), placed, at)
                lead = max([lead, *(k - firing(at) for k, tile in made if meet(window, tile) is not None)])
        return lead

    def _overlapping_split_loops(
        self, transfer: TransferNode, mapping: "Mapping", placed: set[LayerDim], fusion_splits: dict[LayerDim, int]
    ) -> dict[LayerDim, int]:
        """The trips of each fused loop over a dim a split places along which a copy of ``transfer`` reads a window
        overlapping a neighbouring core's tile."""
        dims = [dim for dim in placed if fusion_splits.get(dim, 1) > 1]
        crossed = {dim for copy in transfer.outputs for dim in self.get_windows(copy, transfer, mapping, dims)}
        return {dim: n for dim, n in fusion_splits.items() if dim in crossed}

    def _trip_tiles(self, reads: Sequence[_Read], placed: set[LayerDim], trips: dict[LayerDim, int]) -> list[Box]:
        """The hull of what ``reads`` cover at each trip of the fused loops ``trips``, less what an earlier trip
        covers."""
        tiles: list[Box] = []
        for steps in product(*(range(n) for n in trips.values())):
            tile = _hull([self._read_box(read, placed, dict(zip(trips, steps, strict=True))) for read in reads])
            tiles += less(tile, tiles)
        return tiles

    def _firing_nest(
        self,
        transfer: TransferNode,
        placed: set[LayerDim],
        ssis: Callable[[Node | Tensor], SteadyStateIterationSpace | None],
        fusion_splits: dict[LayerDim, int],
        crossed: set[LayerDim],
    ) -> _FiringNest | None:
        """The fused loops of ``transfer``, where it has several copies or its copy slides: the innermost no split
        places, each one a copy slides along and each placed one a window overlaps a neighbouring core's tile along
        (``crossed``), with each reader's sliding work. None otherwise, one firing then standing for all."""
        if (space := ssis(transfer)) is None:
            return None
        readers = [
            next(n for n in self.successors(transfer) if isinstance(n, HasInputs) and copy in n.inputs)
            for copy in transfer.outputs
        ]
        if not all(isinstance(reader, ComputationNode) for reader in readers):
            return None
        spaces = [copy_space for copy in transfer.outputs if (copy_space := ssis(copy))]
        relevant = {v.dimension for one in (space, *spaces) for v in one.get_temporal_variables() if v.relevant}
        loops = [
            v.dimension
            for v in space.get_temporal_variables()
            if v.dimension in relevant and fusion_splits.get(v.dimension, 1) > 1
        ]
        sliding = {v.dimension for copy_space in spaces for v in copy_space.get_temporal_variables() if v.halo}
        if len(transfer.outputs) == 1 and not (sliding | crossed).intersection(loops):
            return None
        dims: list[LayerDim] = []
        spans: dict[HasIterationSpace, dict[LayerDim, list[tuple[int, int]]]] = {r: {} for r in readers}
        for k, dim in enumerate(dict.fromkeys(loops)):
            if dim in placed:
                dims += [dim] if dim in sliding | crossed else []
                continue
            if k and dim not in sliding:
                continue
            work = self.get_sliding_work(dim, fusion_splits[dim])
            if not all(reader in work for reader in readers):
                continue
            dims.append(dim)
            for reader in readers:
                spans[reader][dim] = self._spans(reader, dim, work[reader])
        if not dims:
            return None
        return _FiringNest(tuple(dims), tuple(fusion_splits[dim] for dim in dims), spans)

    def _spans(self, node: HasIterationSpace, dim: LayerDim, work: tuple[int, ...]) -> list[tuple[int, int]]:
        """The range of ``dim`` ``node`` computes at each step of its sliding ``work`` along its output."""
        unique_dims, dim_values = self.unique_dimensions()
        z = unique_dims.index(dim)
        exprs = [dim_values[g] for g in self.global_idxs[node]]
        rate = next(
            c
            for r in node.get_mapping(node.outputs[0]).results
            if isinstance(r, AffineDimExpr) and (c := affine_coefficients(exprs[r.position], len(unique_dims))[1][z])
        )
        marks = [-1, *(int(end) - 1 for end in np.cumsum(work))]
        return [(ceil((before + 1) / rate), last // rate) for before, last in pairwise(marks)]

    def _target_demand(  # noqa: PLR0913
        self,
        transfer: TransferNode,
        mapping: "Mapping",
        tiling: InterCoreTiling,
        placed: set[LayerDim],
        targets: Sequence[Memory],
        reads_in: Callable[[ComputationNode, Tensor], tuple[Memory, ...]] | None,
        ssis: Callable[[Node | Tensor], SteadyStateIterationSpace | None],
        can_copy_from: Callable[[_Read, _Read], bool] | None,
        *,
        firing: _Firing | None = None,
    ) -> list[_TargetDemand]:
        """Per target, each box the copies read there in a firing (at ``firing`` of the nest), with every copy reading
        it; a box a larger copy in the same memory holds by then (``can_copy_from``) is a local copy from that copy."""
        reads = sorted(
            (
                (c, j, want, read)
                for c, copy in enumerate(transfer.outputs)
                for j, want, read in self._copy_windows(
                    copy, transfer, mapping, tiling, placed, targets, reads_in, firing
                )
            ),
            key=lambda entry: -prod(hi - lo + 1 for lo, hi, _ in entry[2]),
        )
        found: dict[tuple[int, Box], _TargetDemand] = {}
        local_copies: list[_TargetDemand] = []
        for k, (c, j, want, read) in enumerate(reads):
            copied_from = next(
                (
                    other
                    for other, i, _, larger in reads[:k]
                    if can_copy_from is not None
                    and other != c
                    and targets[i] is targets[j]
                    and (firing is None or not less(_plain(want), self._read_so_far(larger, placed, firing)))
                    and can_copy_from(read, larger)
                ),
                None,
            )
            for box in self._new_part(transfer.outputs[c], transfer, mapping, ssis, want, read, placed, firing):
                if copied_from is not None:
                    local_copies.append(_TargetDemand(j, box, want, (c,), True, copied_from))
                    continue
                entry = found.get((j, box), _TargetDemand(j, box, want, (), False))
                found[(j, box)] = replace(entry, copies=(*entry.copies, c))
        return _disjoint_boxes(list(found.values())) + local_copies

    def _read_so_far(self, read: _Read, placed: set[LayerDim], firing: _Firing) -> list[Box]:
        """The windows ``read`` covers at ``firing`` of the nest and at every firing before it."""
        nest, s = firing
        return [_plain(box) for k in range(s + 1) if (box := self._firing_box(read, placed, (nest, k))) is not None]

    def _always_within(
        self, smaller: _Read, larger: _Read, fusion_splits: dict[LayerDim, int], placed: set[LayerDim]
    ) -> bool:
        """Whether the ``larger`` read's window holds the ``smaller`` one's at every trip of the ``fusion_splits``."""
        for steps in product(*(range(n) for n in fusion_splits.values())):
            at = dict(zip(fusion_splits, steps, strict=True))
            if not within(self._read_box(smaller, placed, at), self._read_box(larger, placed, at)):
                return False
        return True

    def _read_box(self, read: _Read, placed: set[LayerDim], steps: dict[LayerDim, int]) -> Box:
        """The box ``read`` covers at the given steps of the fused loops: a step spans a core's tile times the cores
        a split of the same dim spreads it over."""
        tensor, node, split, core, cores = read
        if not isinstance(node, HasIterationSpace):
            return tuple((0, n - 1) for n in tensor.shape)
        at = dict.fromkeys(placed | {d for d, _ in split}, 0) | self._position(split, position_of(core, cores, split))
        factors = dict(split)
        for dim, step in steps.items():
            at[dim] = at.get(dim, 0) + step * factors.get(dim, 1)
        return _plain(self._bounds(tensor, [node], self._tile_sizes(split), at))

    def _demand_box(
        self, tensor: Tensor, node: Node, split: InterCoreTiling, core: int, cores: int, placed: set[LayerDim]
    ) -> list[tuple[int, int, bool]]:
        """The inclusive range per axis of ``tensor`` that ``node`` touches on core ``core`` of ``cores`` under
        ``split``, the dims of ``placed`` at their first tile, and whether a split places the axis."""
        if not isinstance(node, HasIterationSpace):
            return [(0, n - 1, True) for n in tensor.shape]
        at = dict.fromkeys(placed | {d for d, _ in split}, 0) | self._position(split, position_of(core, cores, split))
        return self._bounds(tensor, [node], self._tile_sizes(split), at)

    def _firing_box(
        self, read: _Read, placed: set[LayerDim], firing: _Firing | None
    ) -> list[tuple[int, int, bool]] | None:
        """:meth:`_demand_box` of ``read``; at ``firing`` of the nest, with each of its dims spanning, in tensor
        coordinates, what the reading node computes at that firing, None if it computes nothing then, a placed one at
        its tile."""
        tensor, node, split, core, cores = read
        if firing is None or (spans := firing[0].spans.get(cast(HasIterationSpace, node))) is None:
            return self._demand_box(tensor, node, split, core, cores, placed)
        nest, s = firing
        trips = nest.trip_indices[s]
        ranges = {dim: spans[dim][trips[k]] for k, dim in enumerate(nest.dims) if dim in spans}
        if any(hi < lo for lo, hi in ranges.values()):
            return None
        at = dict.fromkeys(placed | {d for d, _ in split}, 0) | self._position(split, position_of(core, cores, split))
        factors = dict(split)
        for k, dim in enumerate(nest.dims):
            if dim not in spans:
                at[dim] = at.get(dim, 0) + trips[k] * factors.get(dim, 1)
        sizes = self._tile_sizes(split) | {dim: hi - lo + 1 for dim, (lo, hi) in ranges.items()}
        origin = {dim: lo for dim, (lo, _) in ranges.items()}
        return self._bounds(tensor, [cast(HasIterationSpace, node)], sizes, at, origin)

    def _copy_windows(  # noqa: PLR0913
        self,
        copy: Tensor,
        transfer: TransferNode,
        mapping: "Mapping",
        tiling: InterCoreTiling,
        placed: set[LayerDim],
        targets: Sequence[Memory],
        reads_in: Callable[[ComputationNode, Tensor], tuple[Memory, ...]] | None,
        firing: _Firing | None = None,
    ) -> list[tuple[int, list[tuple[int, int, bool]], _Read]]:
        """Each target position of ``copy`` with the window read there (at ``firing`` of the nest) and the read: per
        core of its reader in the memory it reads in (``reads_in``), else per target under the transfer's split."""
        reader = next(n for n in self.successors(transfer) if isinstance(n, HasInputs) and copy in n.inputs)
        if reads_in is not None and isinstance(reader, ComputationNode) and mapping.get(reader).resource_allocation:
            split, memories = self._tiling(reader, mapping), reads_in(reader, copy)
            reads = [
                (j, (copy, reader, split, p, len(memories)))
                for p, memory in enumerate(memories)
                if (j := next((j for j, target in enumerate(targets) if target is memory), None)) is not None
            ]
        else:
            read, node = self._reader(copy, transfer) or (transfer.inputs[0], next(iter(self.predecessors(transfer))))
            reads = [(j, (read, node, tiling, j, len(targets))) for j in range(len(targets))]
        return [
            (j, window, read) for j, read in reads if (window := self._firing_box(read, placed, firing)) is not None
        ]

    def _new_part(  # noqa: PLR0913
        self,
        copy: Tensor,
        transfer: TransferNode,
        mapping: "Mapping",
        ssis: Callable[[Node | Tensor], SteadyStateIterationSpace | None],
        want: list[tuple[int, int, bool]],
        read: _Read,
        placed: set[LayerDim],
        firing: _Firing | None,
    ) -> list[Box]:
        """The part of its window ``want`` that ``copy`` does not hold yet: less what it keeps from the previous trip of
        each loop that slides it, the window then at a firing of the nest over that loop, else the halo along the loop's
        axis."""
        halos = self.sliding_halos(copy, transfer, mapping, ssis(copy))
        nest, s = firing or (None, 0)
        dims = nest.dims if nest else ()
        box = list(_plain(want))
        for dim, (axis, halo) in halos.items():
            lo, hi = box[axis]
            if dim not in dims and hi - lo + 1 > halo:
                box[axis] = (lo + halo, hi)
        before = [
            self._firing_box(read, placed, (nest, s - prod(nest.trips[:k])))
            for k, dim in enumerate(dims)
            if dim in halos and nest.trip_indices[s][k]
        ]
        return less(tuple(box), [_plain(window) for window in before if window is not None])

    def splits_reduction(self, node: HasIterationSpace, tiling: InterCoreTiling) -> bool:
        """Whether ``tiling`` splits a dimension ``node`` reduces over, leaving each core a partial sum."""
        split = {dim for dim, factor in tiling if factor > 1}
        types = derive_iterator_types(node)
        return any(dim in split and types[p] is IteratorType.REDUCTION for p, dim in enumerate(self.get_dims(node)))

    def get_sliding_work(self, dim: LayerDim, splits: int) -> dict[HasIterationSpace, tuple[int, ...]]:
        """How far each node whose loops ``dim`` slides gets along its output in each of the ``splits`` tiles of
        ``dim``: tile i reaches ``h + a*i``, the end of what its readers' tile i reads (their footprints at tiles 0 and
        1) or of its own tile, clipped to the tensor: a lookahead lengthens the first tile, the last ends the tensor."""
        unique_dims, dim_values = self.unique_dimensions()
        sizes = [self.get_dimension_size(z) for z in unique_dims]
        z = unique_dims.index(dim)
        reach: dict[HasIterationSpace, tuple[int, int, dict[int, range], int]] = {}
        work: dict[HasIterationSpace, tuple[int, ...]] = {}
        for node in reversed(self.dataflow_sort()):
            if not isinstance(node, HasIterationSpace):
                continue
            exprs = [dim_values[g] for g in self.global_idxs[node]]
            box = {d: range(lo, hi + 1) for d, (lo, hi) in enumerate(affine_bounds(e, sizes) for e in exprs)}
            output = node.get_mapping(node.outputs[0]).results
            slides = [
                (k, r.position)
                for k, r in enumerate(output)
                if isinstance(r, AffineDimExpr) and affine_coefficients(exprs[r.position], len(sizes))[1][z]
            ]
            if not slides:
                continue
            axis, out = slides[0]
            readers = [c for c in self.successors(node) if c in reach]
            step = affine_coefficients(exprs[out], len(sizes))[1][z] * sizes[z]
            points = [
                max(
                    (
                        footprint(c.get_mapping(t), cb | {at: range(h + a * i, h + a * i + 1)})[axis][-1]
                        for c in readers
                        for h, a, cb, at in [reach[c]]
                        for t in node.outputs
                        if t in c.inputs
                    ),
                    default=box[out][-1] + step * i,
                )
                for i in (0, 1)
            ]
            reach[node] = (points[0], points[1] - points[0], box, out)
            end = node.outputs[0].subview.source.type.get_shape()[axis] - 1
            marks = [-1, *(min(reach[node][0] + reach[node][1] * i, end) for i in range(splits - 1)), end]
            work[node] = tuple(b - a for a, b in pairwise(marks))
        return work

    def get_tensor_of_transfer_from_single_core(
        self, tensor: Tensor, transfer: TransferNode, mapping: "Mapping"
    ) -> Tensor:
        pred_idx = transfer.inputs.index(tensor)
        pred = list(self.predecessors(transfer))[pred_idx]
        if isinstance(pred, InEdge):
            pred_tiling = tuple()
        elif isinstance(pred, TransferNode):
            # A multi-hop staged transfer: its own tiling determines the shape here.
            pred_tiling = self.get_unique_dims_inter_core_tiling(transfer, mapping)
        else:
            assert isinstance(pred, ComputationNode), f"Expected ComputationNode, got {type(pred)}"
            pred_tiling = self.get_unique_dims_inter_core_tiling(pred, mapping)
        index_steps = self._tile_index_steps(tensor, self._accessors(tensor, None, False))
        return self.strided_view(
            self._tile(tensor, self.get_tensor_shape_with_tiling(tensor, pred_tiling), index_steps)
        )

    def with_modified_dimension_sizes(self, new_sizes: dict[LayerDim, int]) -> "Workload":
        """Create a new workload where the dimension sizes of the given global dimension indices are modified to the new
        sizes provided in new_sizes.

        This recreates all tensors (and nodes referencing them) so tensor shapes stay consistent with the updated
        global loop sizes.
        """
        # Infer the updated shape for every tensor based on the strides.
        tensor_map: dict[str, Tensor] = {}
        all_index_steps = self._index_steps_of(new_sizes)
        for node in self.get_computation_nodes():
            for original_tensor in node.tensors:
                nodes = self._accessors(original_tensor, None, False)
                read = all_index_steps.get(original_tensor.subview.source) or _every_index(original_tensor)
                index_steps = self._tile_index_steps(original_tensor, nodes, read)
                shape = self._extents(original_tensor, nodes, new_sizes, {}, index_steps)
                tensor_map[original_tensor.name] = self._tile(original_tensor, shape, index_steps)

        # Recreate nodes in place, so the new workload keeps this one's node order.
        new_nodes: list[Node] = []
        for node in self.nodes:
            if isinstance(node, InEdge):
                # InEdge node name may differ from output tensor name (e.g. Flatten1_in vs flatten_out)
                # Look up by the actual output tensor name first, then fall back to node name
                out_tensor_name = node.outputs[0].name if node.outputs else node.name
                new_output = tensor_map.get(out_tensor_name) or tensor_map.get(node.name)
                assert new_output is not None, (
                    f"InEdge tensor {node.name} (output: {out_tensor_name}) must have been inferred"
                )
                new_node = replace(node, outputs=(new_output,))
            elif isinstance(node, ComputationNode):
                new_inputs = tuple(cast(Tensor, tensor_map[inp.name]) for inp in node.inputs)
                new_output = tensor_map.get(node.outputs[0].name)
                assert new_output is not None, f"ComputationNode output tensor {node.name} must have been inferred"
                # replace() keeps the concrete subclass and its fields (NormalizationNode.reduction_axes).
                new_node = replace(node, inputs=new_inputs, outputs=(new_output,))
            elif isinstance(node, TransferNode):
                new_inputs = tuple(cast(Tensor, tensor_map[inp.name]) for inp in node.inputs)
                new_output = tensor_map.get(node.outputs[0].name)
                assert new_output is not None, f"TransferNode output tensor {node.name} must have been inferred"
                new_node = TransferNode(
                    name=node.name,
                    inputs=new_inputs,
                    outputs=(new_output,),
                    transfer_type=node.transfer_type,
                    operand_mapping=node.operand_mapping,
                )
            elif isinstance(node, FusionEdge):
                # FusionEdge has no iteration space; pass tensors through unchanged
                new_inputs = tuple(cast(Tensor, tensor_map.get(inp.name, inp)) for inp in node.inputs)
                new_outputs = tuple(cast(Tensor, tensor_map.get(out.name, out)) for out in node.outputs)
                new_node = FusionEdge(
                    name=node.name,
                    inputs=new_inputs,
                    outputs=new_outputs,
                    op_type=node.op_type,
                )
            elif isinstance(node, OutEdge):
                new_inputs = tuple(cast(Tensor, tensor_map[inp.name]) for inp in node.inputs)
                new_node = replace(node, inputs=new_inputs)
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            new_nodes.append(new_node)

        return Workload(new_nodes)

    def get_tensor_dimensions(self, tensor: Tensor) -> tuple[LayerDim, ...]:
        """Get all unique LayerDims associated with the given tensor"""
        strides = self.strides_for_tensor(tensor)
        relevant_dims = []
        for dim, stride in strides.items():
            if any(s != 0 for s in stride):
                relevant_dims.append(dim)
        return tuple(relevant_dims)

    def strides_for_tensor(self, tensor: Tensor) -> dict[LayerDim, tuple[int, ...]]:
        unique_dims, dim_values = self.unique_dimensions()
        all_dims = dim_values[: self.num_dims]
        node = next(iter(n for n in self.get_iteration_space_nodes() if tensor in n.tensors))
        mapping = node.get_mapping(tensor)
        global_mapping = self.global_mapping(node, mapping)
        # Baseline: all z = 0
        zero = [0] * len(unique_dims)
        base_all_dims = [int(dim.eval(zero, [])) for dim in all_dims]
        base_out = global_mapping.eval(base_all_dims, [])
        one_list = np.eye(len(unique_dims), dtype=int).tolist()
        result: dict[LayerDim, tuple[int, ...]] = {}
        for unique_dim, var in zip(unique_dims, one_list, strict=True):
            var = cast(list[int], var)
            bumped_all_dims = [int(dim.eval(var, [])) for dim in all_dims]
            bumped_out = global_mapping.eval(bumped_all_dims, [])
            # STRIDE = delta output, constants cancel out
            stride = tuple(int(b) - int(a) for a, b in zip(base_out, bumped_out, strict=True))
            result[unique_dim] = stride
        return result

    _PALETTE = {
        "computation": ("#dbeafe", "#2a78d6"),
        "transfer": ("#ffe6d1", "#eb6834"),
        "in": ("#fdf3c8", "#b9992a"),
        "out": ("#d6f5e3", "#1baf7a"),
        "fusion": ("#ece0ff", "#7d5bd1"),
        "tensor": ("#f1f3f5", "#8a929b"),
        "state": ("#e7f6f0", "#1baf7a"),
    }

    def _kind(self, node: Node) -> str:
        if isinstance(node, ComputationNode):
            return "computation"
        if isinstance(node, TransferNode):
            return "transfer"
        if isinstance(node, InEdge):
            return "in"
        if isinstance(node, OutEdge):
            return "out"
        return "fusion"

    def _node_label(self, node: Node) -> list[str]:
        if isinstance(node, (ComputationNode, TransferNode)):
            dims = {str(d): self.get_dimension_size(d) for d in self.get_dims(node)}
            head = node.name if isinstance(node, ComputationNode) else f"{node.name} · {node.transfer_type.name}"
            return [head, " ".join(f"{k}={v}" for k, v in dims.items())]
        if isinstance(node, InEdge):
            return [node.name, str(node.outputs[0].shape)]
        if isinstance(node, OutEdge):
            return [node.name, str(node.inputs[0].shape)]
        return [node.name, f"[{getattr(node, 'op_type', '')}]"]

    def _tensor_label(self, tensor: Tensor, carried: str | None, mapping, ssis) -> list[str]:
        try:
            dims = {str(d): self.get_dimension_size(d) for d in self.get_tensor_dimensions(tensor)}
        except (StopIteration, KeyError):
            dims = {}
        lines = [tensor.name, str(tensor.shape)]
        if dims:
            lines.append(" ".join(f"{k}={v}" for k, v in dims.items()))
        if carried:
            lines.append(f"resident · carried over {carried}")
        if mapping is not None:
            try:
                allocation = mapping.get(tensor).memory_allocation
                if allocation is not None:
                    lines.append(f"mem {allocation}")
            except KeyError:
                pass
        if ssis:
            loops = self._get_for_loop_label(ssis.get(tensor, None)).strip()
            lines += [x for x in loops.split("\n") if x]
        return lines

    def _carried_over(self) -> dict[str, str]:
        """Tensor name to the dimension it is carried over, for the state a kernel keeps."""
        carried: dict[str, str] = {}
        for node in self.nodes:
            if not isinstance(node, HasIterationSpace):
                continue
            dims = self.get_dims(node)
            for tensor in getattr(node, "inputs", ()):
                if not is_state_operand(node, tensor):
                    continue
                positions = sorted(sequential_dims(node))
                carried[tensor.name] = str(dims[positions[0]]) if positions else "?"
        return carried

    def visualize(
        self,
        filepath: str = "workload_graph.png",
        mapping: "Mapping | None" = None,
        ssis: dict[Node, "SteadyStateIterationSpace"] | None = None,
    ) -> None:
        """Draw the graph, tensors included, as an SVG written beside ``filepath``."""
        carried = self._carried_over()
        boxes: dict[str, tuple[list[str], str]] = {}
        edges: list[tuple[str, str]] = []
        graph = nx.DiGraph()
        # An operation and a tensor may carry the same name, so the two are kept apart.
        for node in self.nodes:
            boxes[f"n:{node.name}"] = (self._node_label(node), self._kind(node))
            graph.add_node(f"n:{node.name}")
        for node in self.nodes:
            for tensor in (*getattr(node, "outputs", ()), *getattr(node, "inputs", ())):
                if f"t:{tensor.name}" not in boxes:
                    boxes[f"t:{tensor.name}"] = (
                        self._tensor_label(tensor, carried.get(tensor.name), mapping, ssis),
                        "state" if tensor.name in carried else "tensor",
                    )
                    graph.add_node(f"t:{tensor.name}")
            for tensor in getattr(node, "outputs", ()):
                edges.append((f"n:{node.name}", f"t:{tensor.name}"))
            for tensor in getattr(node, "inputs", ()):
                edges.append((f"t:{tensor.name}", f"n:{node.name}"))
        graph.add_edges_from(edges)
        target = filepath[: filepath.rfind(".")] + ".svg" if "." in filepath else filepath + ".svg"
        _write_svg(target, boxes, edges, graph, self._PALETTE)

    def _get_for_loop_label(self, ssis: SteadyStateIterationSpace | None) -> str:
        if ssis is not None:
            temporal_loop_dims = reversed(ssis.get_temporal_variables())
            temporal_loop_sizes = reversed(ssis.get_temporal_sizes())
            reuses = reversed(ssis.get_temporal_reuses())
            label = "\nForLoops:"
            indent = ""
            for dim, size, reuse in zip(temporal_loop_dims, temporal_loop_sizes, reuses, strict=True):
                label += f"\n{indent}{dim}: {size}; Reuse={reuse}"
                indent += "  "
            label += "\n"
            return f"{label}"
        return ""

    def get_timeslots(self, mapping: "Mapping | None" = None) -> dict[Node, int]:
        """Assign each node a timeslot using a depthwise priority topological sort (drain a
        transfer chain into its ComputationNode before opening the next branch).

        Slot-sharing rules:
        - A TransferNode and a ComputationNode may always share a slot (disjoint resource
          classes: links vs cores).
        - InEdges and OutEdges have no slot exclusion.
        - When ``mapping`` is provided, two ComputationNodes share a slot iff their
          candidate core allocations admit a pair with disjoint cores; two TransferNodes
          share a slot iff their candidate ``MulticastPathPlan``s admit a pair with
          disjoint ``links_used``. Joint feasibility across all same-class nodes in the
          slot is checked by backtracking, so the downstream constraint solver is
          guaranteed at least one valid resource assignment per slot.
        - When ``mapping`` is None, falls back to ≤1 ComputationNode and ≤1 TransferNode
          per slot.
        """

        position = self.node_positions()

        def priority(node: Node):
            if isinstance(node, ComputationNode):
                return (0, position[node])
            if isinstance(node, TransferNode):
                return (1, position[node])
            if isinstance(node, FusionEdge):
                return (2, position[node])
            if isinstance(node, InEdge):
                return (3, position[node])
            return (4, position[node])  # OutEdge last

        def get_options(node: Node) -> list[frozenset] | None:
            """Return candidate resource sets for a node, or None if unknown.

            For TransferNode: each option is the ``links_used`` of one MulticastPathPlan.
            For ComputationNode: each option is the set of cores in one allocation.
            """
            if mapping is None:
                return None
            try:
                ra = mapping.get(node).resource_allocation
            except (KeyError, AttributeError):
                return None
            if not ra:
                return None
            if isinstance(node, TransferNode):
                return [frozenset(cast("MulticastPathPlan", p).links_used) for p in ra]
            if isinstance(node, ComputationNode):
                return [frozenset(cast("Sequence[Core]", alloc)) for alloc in ra]
            return None

        def joint_feasible(option_lists: list[list[frozenset]]) -> bool:
            """True iff one option from each list can be picked so all are pairwise disjoint."""

            def bt(idx: int, used: frozenset) -> bool:
                if idx == len(option_lists):
                    return True
                for opt in option_lists[idx]:
                    if not (opt & used):
                        if bt(idx + 1, used | opt):
                            return True
                return False

            return bt(0, frozenset())

        def can_join(slot_opts: list[list[frozenset] | None], new_opts: list[frozenset] | None) -> bool:
            if new_opts is None:
                # Unknown allocation: fall back to exclusive use of this resource class.
                return len(slot_opts) == 0
            concrete: list[list[frozenset]] = []
            for o in slot_opts:
                if o is None:
                    return False
                concrete.append(o)
            concrete.append(new_opts)
            return joint_feasible(concrete)

        timeslots: dict[Node, int] = {}
        slot_transfers: dict[int, list[list[frozenset] | None]] = {}
        slot_computes: dict[int, list[list[frozenset] | None]] = {}

        for node in nx.lexicographical_topological_sort(self, key=priority):
            earliest = max((timeslots[p] + 1 for p in self.predecessors(node)), default=0)
            slot = earliest
            bucket: dict[int, list[list[frozenset] | None]] | None = None
            if isinstance(node, TransferNode):
                bucket = slot_transfers
            elif isinstance(node, ComputationNode):
                bucket = slot_computes
            if bucket is not None:
                opts = get_options(node)
                while not can_join(bucket.get(slot, []), opts):
                    slot += 1
                bucket.setdefault(slot, []).append(opts)
            timeslots[node] = slot
        return timeslots

    def _node_ir(self, node: Node) -> dict:
        """Serialize one node: identity plus whatever operand / iteration / type facets it has."""
        node_data: dict = {"name": node.name, "type": type(node).__name__}
        if isinstance(node, HasIterationSpace):
            node_data["dimensions"] = {str(dim): self.get_dimension_size(dim) for dim in self.get_dims(node)}
            node_data["global_dim_indices"] = list(self.global_idxs[node])
        if isinstance(node, HasInputs):
            node_data["inputs"] = [
                {"name": t.name, "shape": list(t.shape), "operand_type": str(t.operand_type)} for t in node.inputs
            ]
        if isinstance(node, HasOutputs):
            node_data["outputs"] = [
                {"name": t.name, "shape": list(t.shape), "operand_type": str(t.operand_type)} for t in node.outputs
            ]
        if isinstance(node, ComputationNode):
            node_data["computation_type"] = str(node.type)
        if isinstance(node, TransferNode):
            node_data["transfer_type"] = str(node.transfer_type)
        if isinstance(node, FusionEdge):
            node_data["fusion_op_type"] = node.op_type
        return node_data

    def get_ir(self) -> dict:
        """Return a dictionary representation of the workload for serialization/inspection.

        This captures:
        - All nodes with their properties
        - All edges between nodes
        - Unique dimensions and their sizes
        - Dimension relationships between layers and unique workload dimensions
        - Tensor information including shapes and strides
        """
        unique_dims, dim_values = self.unique_dimensions()
        unique_dims_info = {
            str(dim): {"index": i, "size": self.get_dimension_size(dim)} for i, dim in enumerate(unique_dims)
        }

        # Build nodes info
        nodes_info = [self._node_ir(node) for node in self.dataflow_sort()]

        # Build edges info
        edges_info = []
        for src, dst in self.edges:
            edge_data = {
                "source": src.name,
                "target": dst.name,
            }
            # Find shared tensor if both have iteration spaces
            if isinstance(src, HasOutputs) and isinstance(dst, HasInputs):
                shared_tensors = [t for t in src.outputs if t in dst.inputs]
                if shared_tensors:
                    edge_data["shared_tensors"] = [t.name for t in shared_tensors]
                edges_info.append(edge_data)

        # Build tensor dimension relationships
        tensor_dim_relations = {}
        for tensor in self.tensors:
            tensor_dims = self.get_tensor_dimensions(tensor)
            strides = self.strides_for_tensor(tensor)
            tensor_dim_relations[tensor.name] = {
                "shape": list(tensor.shape),
                "relevant_dimensions": [str(dim) for dim in tensor_dims],
                "strides_per_dimension": {str(dim): list(stride) for dim, stride in strides.items()},
            }

        # Build dimension relations (constraints between dimensions)
        dim_relations = []
        for expr in self.dimension_relations():
            dim_relations.append(str(expr))

        # Build timeslots
        timeslots = {node.name: slot for node, slot in self.get_timeslots().items()}

        return {
            "num_nodes": len(list(self.nodes)),
            "num_edges": len(list(self.edges)),
            "num_unique_dimensions": len(unique_dims),
            "unique_dimensions": unique_dims_info,
            "dimension_expressions": [str(dv) for dv in dim_values],
            "dimension_relations": dim_relations,
            "nodes": nodes_info,
            "edges": edges_info,
            "tensors": tensor_dim_relations,
            "generations": timeslots,
        }


def position_of(core: int, cores: int, tiling: InterCoreTiling) -> int:
    """The tile of ``tiling`` the core at position ``core`` of ``cores`` holds: its own where each holds one, else
    the cores holding one tile each lie next to each other in core order."""
    tiles = prod(f for _, f in tiling)
    if cores <= tiles:
        return core
    return core // (cores // tiles) if cores % tiles == 0 else core * tiles // cores


def _every_index(tensor: Tensor) -> IndexSteps:
    """Every index of each axis of ``tensor``."""
    return tuple((1, 0) for _ in tensor.subview.source.type.get_shape())


def _stepped_range(low: int, high: int, step: int, first: int) -> tuple[int, int]:
    """The inclusive range of ``k`` for which ``first + step * k`` lies in ``low..high``."""
    return -((first - low) // step), (high - first) // step


def _span(tensor: Tensor) -> tuple[int, ...]:
    """The indices per axis the tile ``tensor`` spans in its buffer, from first to last."""
    strides = tensor.subview.static_strides.get_values()
    return tuple((n - 1) * stride + 1 if n else 0 for n, stride in zip(tensor.shape, strides, strict=True))


def _split(box: list[tuple[int, int]], parts: int) -> list[tuple[tuple[int, int], ...]]:
    """``box`` cut into ``parts`` near-equal boxes along its longest axis."""
    axis = max(range(len(box)), key=lambda a: box[a][1] - box[a][0])
    lo, hi = box[axis]
    edges = [lo + (hi - lo + 1) * k // parts for k in range(parts + 1)]
    return [tuple(box[:axis] + [(edges[k], edges[k + 1] - 1)] + box[axis + 1 :]) for k in range(parts)]


def _over_firings(per_firing: list[list[DemandItem]]) -> list[DemandItem]:
    """The demand of a nest's firings over all of them: firings alike up to a shift are kept once, with the range of
    firings they span as their boxes' first axis; if all are alike, one firing stands for all."""
    classes: dict[tuple, list[int]] = {}
    for s, items in enumerate(per_firing):
        classes.setdefault(_shifted(items), []).append(s)
    if len(classes) == 1:
        return per_firing[0]
    found: list[DemandItem] = []
    start = 0
    for steps in classes.values():
        span = (start, start + len(steps) - 1)
        found += [replace(item, box=(span, *item.box), firings=len(per_firing)) for item in per_firing[steps[0]]]
        start += len(steps)
    return found


def _shifted(items: list[DemandItem]) -> tuple:
    """``items`` with their boxes moved to start at the origin together, in a fixed order."""
    if not items:
        return ()
    origin = [min(item.box[a][0] for item in items) for a in range(len(items[0].box))]
    return tuple(
        sorted(
            (
                replace(item, box=tuple((lo - o, hi - o) for (lo, hi), o in zip(item.box, origin, strict=True)))
                for item in items
            ),
            key=repr,
        )
    )


def _sliding_loops(loops: Sequence[IterationVariable]) -> list[IterationVariable]:
    """The ``loops`` that slide a window: relevant to it, keeping a halo and taking more than one trip."""
    return [v for v in loops if v.relevant and v.halo and v.size > 1]


def _plain(box: list[tuple[int, int, bool]]) -> Box:
    """The ranges of ``box`` without whether each axis is placed."""
    return tuple((lo, hi) for lo, hi, *_ in box)


def _hull(boxes: list[Box]) -> Box:
    """The smallest box holding all of ``boxes``."""
    return tuple((min(lo for lo, _ in axis), max(hi for _, hi in axis)) for axis in zip(*boxes, strict=True))


def _source_part_of_window(have: Box, want: list[tuple[int, int, bool]]) -> Box:
    """What a source holds of ``want`` along the axes a split places; along the others it serves all a window
    reads, producing it over the loop that slides there."""
    return tuple(h if placed else (lo, hi) for h, (lo, hi, placed) in zip(have, want, strict=True))


def _busiest[T](options: list[T], cost: Callable[[T], Any]) -> T:
    """The option that costs most, the first of those as costly."""
    return max(options, key=cost) if len(options) > 1 else options[0]


def _union(boxes: list[list[tuple[int, int]]]) -> int:
    """How many points the inclusive ``boxes`` cover together: inclusion-exclusion, boxes intersecting in boxes."""
    total = 0
    for k in range(1, len(boxes) + 1):
        for subset in combinations(boxes, k):
            sides = (min(b[a][1] for b in subset) - max(b[a][0] for b in subset) + 1 for a in range(len(subset[0])))
            total += (-1) ** (k + 1) * prod(max(0, side) for side in sides)
    return total


def determine_fusion_cut_points(workload: Workload) -> list[str]:
    """Identify node names at which to split the workload into bounded fusion groups.

    Heuristic:
    - **MaxPool front-end boundary:** Each ``MaxPool`` node ends the
      front-end group (Conv -> Relu -> MaxPool).  Splitting after it keeps the
      pooling-core allocation separate from the main residual backbone.
    - **Add+Relu residual boundary:** An ``Add`` node followed by
      exactly one ``Relu`` successor (among ComputationNodes) marks the end of
      a residual block.  The *Relu* is the cut point so that both Add and Relu
      remain in the preceding group and the next Conv starts a new group.
      A fan-out guard ensures that if the Relu feeds more than one
      ComputationNode successor the split is skipped (avoids breaking fan-out
      topology).

    Args:
        workload: The full parsed workload graph.

    Returns:
        An ordered list of node *names* (strings) suitable for passing to
        ``Workload.split_fusion_groups(cut_points=...)``.
    """
    # Collect ComputationNode names in topological order for the "last node" guard
    topo_comp_names: list[str] = []
    for node in workload.dataflow_sort():
        if isinstance(node, ComputationNode):
            topo_comp_names.append(node.name)
    last_comp_name = topo_comp_names[-1] if topo_comp_names else None

    from stream.workload.fusion.analysis import barrier_cut_points  # noqa: PLC0415 -- avoid import cycle

    wanted: set[str] = set(barrier_cut_points(workload))
    for node in workload.dataflow_sort():
        if not isinstance(node, ComputationNode):
            continue

        # MaxPool ends the front-end group
        if node.type == "MaxPool":
            wanted.add(node.name)
            continue

        # Add followed by a single Relu successor -> split after Relu
        if node.type == "Add":
            comp_succs = [s for s in workload.successors(node) if isinstance(s, ComputationNode)]
            if len(comp_succs) == 1 and comp_succs[0].type == "Relu":
                wanted.add(comp_succs[0].name)

    # Emit the wanted cuts in topological order (dedupes barrier + heuristic overlaps).
    cut_points = [name for name in topo_comp_names if name in wanted]

    # Guard: do not split after the last ComputationNode in the workload — that
    # would create an empty trailing group with no ComputationNodes.
    if cut_points and cut_points[-1] == last_comp_name:
        cut_points.pop()

    return cut_points
