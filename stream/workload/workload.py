from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
import sympy as sp
from xdsl.dialects.memref import SubviewOp
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper

from stream.datatypes import InterCoreTiling, LayerDim
from stream.workload._svg import write_svg as _write_svg
from stream.workload.affine_transform import AffineTransform
from stream.workload.iterator_type import is_state_operand, sequential_dims
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
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.tensor import Tensor
from stream.workload.utils import affine_bounds, sympy_to_xdsl

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.mapping.mapping import Mapping


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
    _dataflow_order: list[Node] | None = None
    _global_dimension_idxs: dict[Node, range] | None = None

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

    def _invalidate_order(self) -> None:
        self._dataflow_order = None
        self._global_dimension_idxs = None

    def dataflow_sort(self) -> list[Node]:
        """Nodes in topological order, ties broken by the order they were added.

        The frontend adds nodes in the order the source graph lists them, so ties
        follow the dataflow rather than the node names. Renaming a tensor must not
        renumber the dimensions and solver variables derived from this order.
        """
        if self._dataflow_order is None or len(self._dataflow_order) != self.number_of_nodes():
            position = self.node_positions()
            self._dataflow_order = list(nx.lexicographical_topological_sort(self, key=position.__getitem__))
            self._global_dimension_idxs = None
        return self._dataflow_order

    def node_positions(self) -> dict[Node, int]:
        return {node: i for i, node in enumerate(self.nodes)}

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        node_names = ", ".join(getattr(n, "name", type(n).__name__) for n in nodes)
        return f"Workload(num_nodes={len(nodes)}, num_edges={len(edges)}, nodes=[{node_names}])"

    @property
    def num_dims(self):
        return sum(node.num_dims for node in self.nodes if isinstance(node, HasIterationSpace))

    @property
    def global_idxs(self):
        """
        Determine unique global indices for each dimension in this workload
        """
        order = self.dataflow_sort()
        if self._global_dimension_idxs is None:
            global_dimension_idxs: dict[Node, range] = {}
            idx = 0
            for node in order:
                if isinstance(node, HasIterationSpace):
                    global_dimension_idxs[node] = range(idx, idx + node.num_dims)
                    idx += node.num_dims
            self._global_dimension_idxs = global_dimension_idxs
        return self._global_dimension_idxs

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
        """Whether a dimension relation merges two dims that are the *same* iteration axis.

        A relation is kept for the global dedup only when it has the clean form ``d_a - d_b (+ const)``
        -- exactly two dim terms with coefficients ``+1`` and ``-1`` (a constant padding offset is
        allowed). That is a genuine identity: the producer axis and the consumer axis are one and the
        same, so they collapse to a single ``LayerDim``.

        Windowed / strided couplings -- a conv or pool input index ``2*ox + fx``, which yields a
        coefficient other than ``+/-1`` or more than two dim terms -- are cross-node *dependencies*,
        not identities. Folding them into the global equality system is what forced output-spatial
        dims to become compound (untileable) expressions. They are excluded here so every node keeps
        pure, tileable iteration dims; the exact producer region a consumer tile needs (the halo) is
        derived on demand from the affine maps via ``compose_dependency``, not baked into this basis.
        """
        row = AffineTransform.from_affine_map(AffineMap(self.num_dims, 0, (relation,))).A[0]
        return sorted(int(c) for c in row if c != 0) == [-1, 1]

    def dimension_relations(self) -> Sequence[AffineExpr]:
        result = []
        # Relations between shared intermediate tensors:
        for src, dst in self.edges:
            if isinstance(src, HasIterationSpace) and isinstance(dst, HasIterationSpace):
                try:
                    output = next(t for t in src.outputs if t in dst.inputs)
                except StopIteration as e:
                    raise RuntimeError(f"No shared tensor between nodes {src.name} and {dst.name}") from e
                mapping_out = self.global_mapping(src, src.get_mapping(output))
                mapping_in = self.global_mapping(dst, dst.get_mapping(output))
                for expr_out, expr_in in zip(mapping_out.results, mapping_in.results, strict=True):
                    # expr_out == expr_in <=> expr_out - expr_in == 0; keep only identity merges.
                    relation = expr_out - expr_in
                    if self._is_identity_relation(relation):
                        result.append(relation)
        # Relations between shared inputs:
        for node in self.nodes:
            if isinstance(node, InEdge):
                assert len(node.outputs) == 1, "Only single output InEdge supported for now."
                output = node.outputs[0]
                all_users = [cast(HasIterationSpace, out) for (_, out) in self.out_edges(node)]
                for a, b in combinations(all_users, 2):
                    mapping_a = self.global_mapping(a, a.get_mapping(output))
                    mapping_b = self.global_mapping(b, b.get_mapping(output))
                    for expr_a, expr_b in zip(mapping_a.results, mapping_b.results, strict=True):
                        relation = expr_a - expr_b
                        if self._is_identity_relation(relation) and not self._both_parallel_outputs(
                            a, b, expr_a, expr_b
                        ):
                            result.append(relation)
        return result

    def _both_parallel_outputs(
        self, a: "HasIterationSpace", b: "HasIterationSpace", expr_a: AffineExpr, expr_b: AffineExpr
    ) -> bool:
        """Whether a shared-input axis is a PARALLEL output for both consumers ``a`` and ``b``."""
        from stream.workload.iterator_type import IteratorType, derive_iterator_types  # noqa: PLC0415

        if not (isinstance(expr_a, AffineDimExpr) and isinstance(expr_b, AffineDimExpr)):
            return False
        a_local = expr_a.position - self.global_idxs[a].start
        b_local = expr_b.position - self.global_idxs[b].start
        types_a = derive_iterator_types(a)
        types_b = derive_iterator_types(b)
        return types_a.get(a_local) == IteratorType.PARALLEL and types_b.get(b_local) == IteratorType.PARALLEL

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

    def get_fusion_edges(self) -> tuple[FusionEdge, ...]:
        return tuple(cast(FusionEdge, node) for node in self.nodes if isinstance(node, FusionEdge))

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
        fusion_edges = [n for n in self.nodes if isinstance(n, FusionEdge)]
        cut_point_set: set[str] = set(cut_points) if cut_points else set()
        if not fusion_edges and not cut_point_set:
            return [self]

        # Assign each non-FusionEdge, non-InEdge node to a group index.
        # Group boundaries are defined by FusionEdge nodes.
        topo_order = self.dataflow_sort()

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
        result_to_shape: list[tuple[AffineExpr, int]] = []
        for node in self.get_iteration_space_nodes():
            for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
                global_mapping = self.global_mapping(node, mapping)
                for expr, sz in zip(global_mapping.results, tensor.shape, strict=True):
                    result_to_shape.append((expr, sz))

        # Step 1: direct read for dims that appear as pure AffineDimExpr
        dim_to_size: dict[int, int] = {}
        for expr, sz in result_to_shape:
            if isinstance(expr, AffineDimExpr) and expr.position not in dim_to_size:
                dim_to_size[expr.position] = sz

        # Step 2: infer size for remaining dims (kernel dims in strided ops like MaxPool)
        # These dims only appear in affine expressions like: stride*other_dim + kernel_dim + offset
        # Their range is derived from: tensor_size, stride, output_size, and offset
        missing = sorted(set(range(self.num_dims)) - set(dim_to_size.keys()))
        for missing_dim in missing:
            for expr, sz in result_to_shape:
                probe = [0] * self.num_dims
                at_0 = int(expr.eval(probe, []))
                probe[missing_dim] = 1
                at_1 = int(expr.eval(probe, []))
                coeff = at_1 - at_0
                if coeff == 0:
                    continue  # this dim does not appear in this expression
                # Compute max contribution from all other known dims
                other_max = 0
                for d, dsize in dim_to_size.items():
                    probe2 = [0] * self.num_dims
                    probe2[d] = dsize - 1
                    val_hi = int(expr.eval(probe2, []))
                    probe2[d] = 0
                    val_lo = int(expr.eval(probe2, []))
                    if val_hi > val_lo:
                        other_max += val_hi - val_lo
                const_term = int(expr.eval([0] * self.num_dims, []))
                d_i_max = (sz - 1 - const_term - other_max) // coeff
                dim_to_size[missing_dim] = int(d_i_max) + 1
                break

        assert len(dim_to_size) == self.num_dims, (
            f"Could not determine sizes for all {self.num_dims} dims: "
            f"missing {sorted(set(range(self.num_dims)) - set(dim_to_size.keys()))}"
        )
        return tuple(dim_to_size[i] for i in range(self.num_dims))

    def get_dims(self, node: HasIterationSpace) -> list[LayerDim]:
        global_idxs = self.global_idxs
        _, expressions = self.unique_dimensions()
        start_idx = global_idxs[node].start
        stop_idx = global_idxs[node].stop
        dims = expressions[start_idx:stop_idx]
        return dims

    def get_dimension_size(self, dim: LayerDim) -> int:
        _, expressions = self.unique_dimensions()
        dim_ranges = self.get_dimension_sizes()
        idx = expressions.index(dim)
        return dim_ranges[idx]

    def unique_dimensions(self):
        relations = AffineMap(self.num_dims, 0, tuple(self.dimension_relations()))
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

        dim_values = [sympy_to_xdsl(sp.simplify(expr)) for expr in x]
        z = [LayerDim(position=i, prefix="z") for i in range(len(free_vars))]
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
            if "z" in str(dim):
                unique_dim = dim
            else:
                dim_idx = dim.position
                unique_dim = unique_node_dims[dim_idx]
            converted_tiling.append((unique_dim, factor))
        return tuple(converted_tiling)

    def get_tensor_shape_with_dimension_sizes(
        self, tensor: Tensor, dimension_sizes: dict[LayerDim, int]
    ) -> tuple[int, ...]:
        unique_dims, dim_values = self.unique_dimensions()
        # The size of each unique dim z0..zN
        z_sizes = [dimension_sizes[z] for z in unique_dims]
        # This is the logical tensor domain we want to clip to.
        # If your Tensor already knows its shape, use it.
        logical_shape = tensor.shape  # type: ignore[attr-defined]

        def extents_for(node: HasIterationSpace) -> list[int]:
            global_mapping = self.global_mapping(node, node.get_mapping(tensor))
            out: list[int] = []
            for axis, idx_expr in enumerate(global_mapping.results):
                idx_expr_in_z = idx_expr.replace_dims_and_symbols(dim_values, ())
                amin, amax = affine_bounds(idx_expr_in_z, z_sizes)
                # Clip to valid tensor index range [0, logical_shape[axis]-1]
                lo = max(amin, 0)
                hi = min(amax, logical_shape[axis] - 1)
                out.append(int(max(0, hi - lo + 1)))
            return out

        # A tensor several nodes access can have a different footprint per accessor; take the largest.
        shapes = [extents_for(n) for n in self.get_iteration_space_nodes() if tensor in n.tensors]
        return tuple(max(axis_extents) for axis_extents in zip(*shapes, strict=True))

    def get_tensor_shape_with_tiling(self, tensor: Tensor, tiling: InterCoreTiling) -> tuple[int, ...]:
        unique_dims, _ = self.unique_dimensions()
        dim_sizes = {}
        for dim in unique_dims:
            if any(dim == ict[0] for ict in tiling):
                tiling_factor = next(ict[1] for ict in tiling if dim == ict[0])
                dim_size = self.get_dimension_size(dim) // tiling_factor
            else:
                dim_size = self.get_dimension_size(dim)
            dim_sizes[dim] = dim_size
        new_shape = self.get_tensor_shape_with_dimension_sizes(tensor, dim_sizes)
        return new_shape

    def get_tensor_single_core(self, tensor: Tensor, node: HasOutputs, mapping: "Mapping") -> Tensor:
        """
        Get a new Tensor representing the portion residing on a single core, based on the nodes' tiling.
        """
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        tilings = node_mapping.inter_core_tiling
        if not tilings:
            return tensor
        # Assert all possible tilings are equal for now and take first one
        assert all(t == tilings[0] for t in tilings), "Multiple different tilings not implemented yet."
        tiling = tilings[0]
        if tiling == tuple():
            return tensor
        new_shape = self.get_tensor_shape_with_tiling(tensor, tiling)
        new_subview = SubviewOp.from_static_parameters(
            source=tensor.subview.source,
            source_type=tensor.subview.source.type,
            offsets=[0 for _ in new_shape],
            sizes=new_shape,
            strides=[1 for _ in new_shape],
        )
        return Tensor(
            name=tensor.name,
            operand_type=tensor.operand_type,
            shape=new_shape,
            subview=new_subview,
        )

    def get_tensor_of_transfer_to_single_core(
        self, tensor: Tensor, transfer: TransferNode, mapping: "Mapping"
    ) -> Tensor:
        succ_idx = transfer.outputs.index(tensor)
        succ = list(self.successors(transfer))[succ_idx]
        if isinstance(succ, OutEdge):
            tiling = tuple()
        elif isinstance(succ, TransferNode):
            # Current transfer's tiling determines the shape
            tiling = self.get_unique_dims_inter_core_tiling(transfer, mapping)
        elif isinstance(succ, ComputationNode):
            tiling = self.get_unique_dims_inter_core_tiling(succ, mapping)
        else:
            raise TypeError(f"Unexpected successor type {type(succ)} for transfer node {transfer.name}")
        new_shape = self.get_tensor_shape_with_tiling(tensor, tiling)
        new_subview = SubviewOp.from_static_parameters(
            source=tensor.subview.source,
            source_type=tensor.subview.source.type,
            offsets=[0 for _ in new_shape],
            sizes=new_shape,
            strides=[1 for _ in new_shape],
        )
        return Tensor(
            name=tensor.name,
            operand_type=tensor.operand_type,
            shape=new_shape,
            subview=new_subview,
        )

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
        new_shape = self.get_tensor_shape_with_tiling(tensor, pred_tiling)
        new_subview = SubviewOp.from_static_parameters(
            source=tensor.subview.source,
            source_type=tensor.subview.source.type,
            offsets=[0 for _ in new_shape],
            sizes=new_shape,
            strides=[1 for _ in new_shape],
        )
        return Tensor(
            name=tensor.name,
            operand_type=tensor.operand_type,
            shape=new_shape,
            subview=new_subview,
        )

    def replace_node(self, old_node: Node, new_node: Node) -> None:
        """Replace a node in the workload with a new node, updating edges accordingly."""
        if old_node not in self.node_list:
            try:
                old_node = self.get_node_by_name(old_node.name)
            except KeyError as e:
                raise KeyError(f"Node {old_node.name} not found in workload.") from e
        self.add_node(new_node)
        for pred in self.predecessors(old_node):
            self.add_edge(pred, new_node)
        for succ in self.successors(old_node):
            self.add_edge(new_node, succ)
        self.remove_node(old_node)
        self._invalidate_order()

    def with_modified_dimension_sizes(self, new_sizes: dict[LayerDim, int]) -> "Workload":
        """Create a new workload where the dimension sizes of the given global dimension indices are modified to the new
        sizes provided in new_sizes.

        This recreates all tensors (and nodes referencing them) so tensor shapes stay consistent with the updated
        global loop sizes.
        """
        # Infer the updated shape for every tensor based on the strides.
        inferred_shapes: dict[str, tuple[int, ...]] = {}
        original_tensors_dict: dict[str, Tensor] = {}
        for node in self.get_computation_nodes():
            original_tensors = node.tensors
            for original_tensor in original_tensors:
                tensor_name = original_tensor.name
                original_tensors_dict[tensor_name] = original_tensor
                new_shape_t = self.get_tensor_shape_with_dimension_sizes(original_tensor, new_sizes)
                inferred_shapes[tensor_name] = new_shape_t

        # Create new Tensor objects with the inferred shapes.
        tensor_map: dict[str, Tensor] = {}
        for tensor_name, new_shape_t in inferred_shapes.items():
            original_tensor = original_tensors_dict[tensor_name]
            original_subview = original_tensor.subview
            # Create new subview referencing original one with new sizes
            new_subview = SubviewOp.from_static_parameters(
                source=original_subview.source,
                source_type=original_subview.source.type,
                offsets=[0 for _ in new_shape_t],
                sizes=new_shape_t,
                strides=[1 for _ in new_shape_t],
            )
            new_output = Tensor(
                name=tensor_name,
                operand_type=original_tensor.operand_type,
                shape=new_shape_t,
                subview=new_subview,
            )
            tensor_map[tensor_name] = new_output

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
                new_node = InEdge(
                    name=node.name,
                    outputs=(new_output,),
                )
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
                new_node = OutEdge(
                    name=node.name,
                    inputs=new_inputs,
                )
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
        unique_dims, all_dims = self.unique_dimensions()
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

    def _get_mem_alloc_label(self, node: TransferNode, mapping: "Mapping | None") -> str:
        if mapping is not None:
            node_mapping = mapping.get(node)
            if node_mapping.memory_allocation is not None:
                return f"\nMemAlloc: {node_mapping.memory_allocation}"
        return ""

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

    def get_timeslots_simple(self) -> dict[Node, int]:
        """Original baseline: walk topological generations and give every node its own
        unique slot (slot increments per node, not per generation). Kept for A/B-comparison
        against the resource-aware ``get_timeslots``.
        """
        timeslots: dict[Node, int] = {}
        slot = 0
        for generation in nx.topological_generations(self):
            for node in generation:
                timeslots[node] = slot
                slot += 1
        return timeslots

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
        dim_sizes = self.get_dimension_sizes()

        # Build unique dimensions info. `dim_sizes` is indexed by *global loop slot* (0..num_dims-1),
        # while `unique_dims` are the free variables z0..zk -- a unique dim z_i does NOT live at global
        # slot i. Its size is the size of the loop slot that *is* that free variable, i.e. the slot j
        # where dim_values[j] == z_i (that slot always exists: it is the free variable's own column).
        # This keeps unique_dimensions consistent with each node's per-dimension sizes.
        value_strs = [str(dv) for dv in dim_values]
        unique_dims_info = {}
        for i, dim in enumerate(unique_dims):
            key = str(dim)
            slot = value_strs.index(key) if key in value_strs else None
            unique_dims_info[key] = {
                "index": i,
                "size": dim_sizes[slot] if slot is not None and slot < len(dim_sizes) else None,
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
