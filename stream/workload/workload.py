from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
import sympy as sp
from networkx.drawing.nx_pydot import to_pydot  # type: ignore
from snaxc.ir.dart.affine_transform import AffineTransform
from xdsl.dialects.memref import SubviewOp
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper

from stream.datatypes import InterCoreTiling, LayerDim
from stream.workload.node import (
    ComputationNode,
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
    from stream.mapping.mapping import Mapping


class Workload(DiGraphWrapper[Node]):
    def __init__(self, nodes: Sequence[Node] = ()):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for node in nodes:
            if isinstance(node, HasInputs):
                for input in node.inputs:
                    try:
                        pred = next(n for n in nodes if isinstance(n, HasOutputs) and input in n.outputs)
                    except StopIteration as e:
                        raise RuntimeError(f"Input tensor {input.name} for node {node.name} has no producer.") from e
                    graph.add_edge(pred, node)
        super().__init__(graph)

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
        global_dimension_idxs: dict[Node, range] = {}
        idx = 0
        for node in nx.lexicographical_topological_sort(self, key=lambda node: node.name):
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
                    # expr_out == expr_in <=> expr_out - expr_in == 0
                    result.append(expr_out - expr_in)
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
                        result.append(expr_a - expr_b)
        return result

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

    def get_dimension_sizes(self) -> tuple[int, ...]:
        results = []
        shapes: list[int] = []
        for node in self.get_iteration_space_nodes():
            for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
                global_mapping = self.global_mapping(node, mapping)
                results.extend(global_mapping.results)
                shapes.extend(tensor.shape)
        total_map = AffineMap(self.num_dims, 0, tuple(results))
        return total_map.inverse_permutation().eval(shapes, [])

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
        rref_A, pivots = A_sp.rref()

        n_vars = transform.A.shape[1]
        free_vars = [i for i in range(n_vars) if i not in pivots]

        basis_vectors = []
        for free in free_vars:
            v = sp.zeros(n_vars, 1)
            v[free] = 1
            for row, pivot in enumerate(pivots):
                v[pivot] = -rref_A[row, free]
            basis_vectors.append(v)

        N = sp.Matrix.hstack(*basis_vectors)
        z_syms = sp.symbols(f"z0:{len(free_vars)}")
        x = N * sp.Matrix(z_syms)

        dim_values = [sympy_to_xdsl(sp.simplify(expr)) for expr in x]
        # IMPORTANT: z dimensions are LayerDim, so expressions.index(LayerDim(i)) works
        z = [LayerDim(position=i, prefix="z") for i in range(len(free_vars))]
        return z, dim_values

    def get_unique_dims_inter_core_tiling(self, node: ComputationNode, mapping: "Mapping") -> InterCoreTiling:
        """Convert inter_core_tiling dimensions from LayerDim to unique workload indices."""
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        unique_node_dims = self.get_dims(node)
        converted_tiling: list[InterCoreTiling] = []
        all_tilings_equal = all(t == node_mapping.inter_core_tiling[0] for t in node_mapping.inter_core_tiling)
        assert all_tilings_equal, f"Multiple different inter-core tilings for node {node.name} not supported for now."
        for dim, factor in node_mapping.inter_core_tiling[0]:
            if "z" in str(dim):
                unique_dim = dim
            else:
                dim_idx = dim.position
                unique_dim = unique_node_dims[dim_idx]
            converted_tiling.append((unique_dim, factor))
        return converted_tiling

    def get_tensor_shape_with_dimension_sizes(
        self, tensor: Tensor, dimension_sizes: dict[LayerDim, int]
    ) -> tuple[int, ...]:
        unique_dims, dim_values = self.unique_dimensions()
        # The size of each unique dim z0..zN
        z_sizes = [dimension_sizes[z] for z in unique_dims]
        node = next(iter(n for n in self.get_iteration_space_nodes() if tensor in n.tensors))
        mapping = node.get_mapping(tensor)
        global_mapping = self.global_mapping(node, mapping)
        # This is the logical tensor domain we want to clip to.
        # If your Tensor already knows its shape, use it.
        logical_shape = tensor.shape  # type: ignore[attr-defined]
        out_shape: list[int] = []
        for axis, idx_expr in enumerate(global_mapping.results):
            idx_expr_in_z = idx_expr.replace_dims_and_symbols(dim_values, ())
            amin, amax = affine_bounds(idx_expr_in_z, z_sizes)
            # Clip to valid tensor index range [0, logical_shape[axis]-1]
            lo = max(amin, 0)
            hi = min(amax, logical_shape[axis] - 1)
            extent = max(0, hi - lo + 1)
            out_shape.append(int(extent))
        return tuple(out_shape)

    def get_tensor_shape_with_tiling(self, tensor: Tensor, succ_tiling: InterCoreTiling) -> tuple[int, ...]:
        unique_dims, _ = self.unique_dimensions()
        dim_sizes = {}
        for dim in unique_dims:
            if any(dim == ict[0] for ict in succ_tiling):
                tiling_factor = next(ict[1] for ict in succ_tiling if dim == ict[0])
                dim_size = self.get_dimension_size(dim) // tiling_factor
            else:
                dim_size = self.get_dimension_size(dim)
            dim_sizes[dim] = dim_size
        new_shape = self.get_tensor_shape_with_dimension_sizes(tensor, dim_sizes)
        return new_shape

    def get_tensor_of_transfer_to_single_core(
        self, tensor: Tensor, transfer: TransferNode, mapping: "Mapping"
    ) -> Tensor:
        succ_idx = transfer.outputs.index(tensor)
        succ = list(self.successors(transfer))[succ_idx]
        if isinstance(succ, OutEdge):
            succ_tiling = tuple()
        elif isinstance(succ, TransferNode):
            # Successor transfer node should have same tiling as current transfer since they are on the same core
            succ_tiling = self.get_unique_dims_inter_core_tiling(succ, mapping)
        elif isinstance(succ, ComputationNode):
            succ_tiling = self.get_unique_dims_inter_core_tiling(succ, mapping)
        else:
            raise TypeError(f"Unexpected successor type {type(succ)} for transfer node {transfer.name}")
        new_shape = self.get_tensor_shape_with_tiling(tensor, succ_tiling)
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

        # Recreate nodes in topological order so inputs are available when recreating a consumer.
        new_nodes: list[Node] = []
        for node in nx.lexicographical_topological_sort(self, key=lambda n: n.name):
            if isinstance(node, InEdge):
                new_output = tensor_map.get(node.name)
                assert new_output is not None, f"InEdge tensor {node.name} must have been inferred"
                new_node = InEdge(
                    name=node.name,
                    outputs=(new_output,),
                )
            elif isinstance(node, ComputationNode):
                new_inputs = tuple(cast(Tensor, tensor_map[inp.name]) for inp in node.inputs)
                new_output = tensor_map.get(node.outputs[0].name)
                assert new_output is not None, f"ComputationNode output tensor {node.name} must have been inferred"
                new_node = ComputationNode(
                    type=node.type,
                    name=node.name,
                    inputs=new_inputs,
                    outputs=(new_output,),
                    operand_mapping=node.operand_mapping,
                )
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

    def visualize(
        self,
        filepath: str = "workload_graph.png",
        mapping: "Mapping | None" = None,
        ssis: dict[Node, "SteadyStateIterationSpace"] | None = None,
    ) -> None:
        """Visualize the graph using Graphviz and save it to an image file.

        Nodes are laid out horizontally by topological generation,
        and vertically stacked to avoid overlap. The resource is displayed
        below each node in a clean and readable way.
        """
        dot = to_pydot(self)

        # Set global graph layout left to right
        dot.set_rankdir("LR")
        dot.set_concentrate(True)

        # Determine node positions based on topological generations
        generation_to_nodes = {}
        for gen_idx, generation in enumerate(nx.topological_generations(self)):
            generation_to_nodes[gen_idx] = list(generation)

        # Assign nodes to horizontal positions based on their generation
        for gen_idx, nodes in generation_to_nodes.items():
            for idx, node in enumerate(nodes):
                n = dot.get_node(str(node))[0]
                n.set_pos(f"{gen_idx},{-idx}!")  # Horizontal by generation, vertical by index

        # Customize node appearances and add resource labels
        for node in self.nodes():
            n = dot.get_node(str(node))[0]
            if isinstance(node, ComputationNode):
                dim_sizes = {str(dim): self.get_dimension_size(dim) for dim in self.get_dims(node)}
                n.set_shape("ellipse")
                n.set_label(f"{node.name}\nDims: {dim_sizes}")
                n.set_style("filled")
                n.set_fillcolor("#60bcf0")
            elif isinstance(node, TransferNode):
                dim_sizes = {str(dim): self.get_dimension_size(dim) for dim in self.get_dims(node)}
                n.set_shape("box")
                label = f"{node.name}\nType: {node.transfer_type}\nDims: {dim_sizes}"
                label += self._get_mem_alloc_label(node, mapping)
                if ssis:
                    label += self._get_for_loop_label(ssis.get(node, None))

                n.set_label(label)
                n.set_style("filled")
                n.set_fillcolor("#ffcb9a")
            elif isinstance(node, InEdge):
                n.set_shape("box")
                n.set_label(f"{node.name}\nShape: {node.outputs[0].shape}")
                n.set_style("filled")
                n.set_fillcolor("#eaff76e1")
            elif isinstance(node, OutEdge):
                n.set_shape("box")
                n.set_label(f"{node.name}\nShape: {node.inputs[0].shape}")
                n.set_style("filled")
                n.set_fillcolor("#c2f0c2")
            else:
                raise ValueError(f"Unknown node type: {type(node)}")

        # Save to file
        dot.write_png(filepath)
        print(f"Graph saved to {filepath}")

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

    def get_timeslots(self) -> dict[Node, int]:
        timeslots = {}
        slot = 0
        for generation in nx.topological_generations(self):
            for node in generation:
                timeslots[node] = slot
                slot += 1
        return timeslots

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

        # Build unique dimensions info
        unique_dims_info = {
            str(dim): {
                "index": i,
                "size": dim_sizes[i] if i < len(dim_sizes) else None,
            }
            for i, dim in enumerate(unique_dims)
        }

        # Build nodes info
        nodes_info = []
        for node in nx.lexicographical_topological_sort(self, key=lambda n: n.name):
            node_data: dict = {
                "name": node.name,
                "type": type(node).__name__,
            }

            if isinstance(node, HasIterationSpace):
                node_dims = self.get_dims(node)
                node_data["dimensions"] = {str(dim): self.get_dimension_size(dim) for dim in node_dims}
                node_data["global_dim_indices"] = list(self.global_idxs[node])

            if isinstance(node, HasInputs):
                node_data["inputs"] = [
                    {
                        "name": t.name,
                        "shape": list(t.shape),
                        "operand_type": str(t.operand_type),
                    }
                    for t in node.inputs
                ]

            if isinstance(node, HasOutputs):
                node_data["outputs"] = [
                    {
                        "name": t.name,
                        "shape": list(t.shape),
                        "operand_type": str(t.operand_type),
                    }
                    for t in node.outputs
                ]

            if isinstance(node, ComputationNode):
                node_data["computation_type"] = str(node.type)

            if isinstance(node, TransferNode):
                node_data["transfer_type"] = str(node.transfer_type)

            nodes_info.append(node_data)

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
        timeslots = {
            node.name: gen_id
            for gen_id, generation in enumerate(nx.topological_generations(self))
            for node in generation
        }

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
