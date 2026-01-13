from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Flag
from itertools import combinations
from math import prod
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
import sympy as sp
from networkx.drawing.nx_pydot import to_pydot  # type: ignore
from snaxc.ir.dart.affine_transform import AffineTransform
from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.dialects.memref import SubviewOp
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper

from stream.datatypes import InterCoreTiling, LayerDim

if TYPE_CHECKING:
    from stream.mapping.mapping import Mapping


@dataclass(frozen=True, repr=False)
class Tensor:
    name: str
    operand_type: FixedBitwidthType
    shape: tuple[int, ...]
    subview: SubviewOp

    def __repr__(self):
        return f"Tensor(name={self.name}, operand_type={self.operand_type}, shape={self.shape})"

    def size_elements(self) -> int:
        return prod(self.shape)

    def size_bits(self) -> int:
        return self.operand_type.bitwidth * self.size_elements()


@dataclass(frozen=True, repr=False)
class Node(ABC):
    name: str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


@dataclass(frozen=True, repr=False)
class HasOutputs(Node, ABC):
    outputs: tuple[Tensor, ...]

    @property
    def output(self) -> Tensor:
        return self.outputs[0]


@dataclass(frozen=True, repr=False)
class HasInputs(Node, ABC):
    inputs: tuple[HasOutputs, ...]


@dataclass(frozen=True, repr=False)
class InEdge(HasOutputs): ...


@dataclass(frozen=True, repr=False)
class OutEdge(HasInputs): ...


class TransferType(Flag):
    """Flags for different types of data transfer operations (can be combined)."""

    NONE = 0
    UNICAST = 1
    DISTRIBUTE = 2
    BROADCAST = 3
    JOIN = 4
    REDUCE = 5


@dataclass(frozen=True, repr=False)
class HasIterationSpace(HasInputs, HasOutputs):
    operand_mapping: tuple[AffineMap, ...]

    @property
    def num_dims(self) -> int:
        # Dimensionality of all maps should be equal
        return self.operand_mapping[0].num_dims

    def get_mapping(self, operand: Node | Tensor) -> AffineMap:
        if operand is self.output:
            return self.operand_mapping[-1]
        for i, input in enumerate(self.inputs):
            if operand is input or operand is input.output:
                return self.operand_mapping[i]
        raise RuntimeError

    def get_dimension_size(self, layer_dim: LayerDim) -> int:
        dim_index = layer_dim.get_idx()
        return self.output.shape[dim_index]  # TODO: Probably not always of output tensor

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return tuple(inp.output for inp in self.inputs) + (self.output,)


@dataclass(frozen=True, repr=False)
class TransferNode(HasIterationSpace):
    transfer_type: TransferType


@dataclass(frozen=True, repr=False)
class ComputationNode(HasIterationSpace):
    def has_same_performance(self, other: "ComputationNode") -> bool:
        """Check if this computation node has the same performance characteristics as another node.
        This is a simple check based on operand data types and shapes.
        More sophisticated checks may be needed in the future."""
        if len(self.inputs) != len(other.inputs):
            return False
        for inp_self, inp_other in zip(self.inputs, other.inputs, strict=True):
            if inp_self.output.operand_type != inp_other.output.operand_type:
                return False
            if inp_self.output.shape != inp_other.output.shape:
                return False
        if self.output.operand_type != other.output.operand_type:
            return False
        if self.output.shape != other.output.shape:
            return False
        return True


class Workload(DiGraphWrapper[Node]):
    def __init__(self, nodes: Sequence[Node] = ()):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for node in nodes:
            if isinstance(node, HasInputs):
                for input in node.inputs:
                    assert input in graph.nodes(), f"Input {input} not in graph nodes"
                    graph.add_edge(input, node)
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
        for edge in self.edges:
            if isinstance(edge[0], HasIterationSpace) and isinstance(edge[1], HasIterationSpace):
                mapping_out = self.global_mapping(edge[0], edge[0].get_mapping(edge[0].output))
                mapping_in = self.global_mapping(edge[1], edge[1].get_mapping(edge[0]))
                for expr_out, expr_in in zip(mapping_out.results, mapping_in.results, strict=True):
                    # expr_out == expr_in <=> expr_out - expr_in == 0
                    result.append(expr_out - expr_in)
        # Relations between shared inputs:
        for node in self.nodes:
            if isinstance(node, InEdge):
                all_users = [cast(HasIterationSpace, out) for (_, out) in self.out_edges(node)]
                for a, b in combinations(all_users, 2):
                    mapping_a = self.global_mapping(a, a.get_mapping(node))
                    mapping_b = self.global_mapping(b, b.get_mapping(node))
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
        for node in self.nodes:
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
            operands = tuple(inp.output for inp in node.inputs) + node.outputs
            for operand, mapping in zip(operands, node.operand_mapping, strict=True):
                global_mapping = self.global_mapping(node, mapping)
                results.extend(global_mapping.results)
                shapes.extend(operand.shape)
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
        unique_dims, expressions = self.unique_dimensions()
        assert dim in unique_dims, "Dimension not found in workload"
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
        z = sp.symbols(f"z0:{len(free_vars)}")
        x = N * sp.Matrix(z)

        dim_values = []
        for expr in x:
            dim_values.append(sp.simplify(expr))

        return z, dim_values

    def get_unique_dims_inter_core_tiling(self, node: ComputationNode, mapping: "Mapping") -> InterCoreTiling:
        """Convert inter_core_tiling dimensions from LayerDim to unique workload indices."""
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        unique_node_dims = self.get_dims(node)
        converted_tiling: InterCoreTiling = []
        for dim, factor in node_mapping.inter_core_tiling:
            dim_idx = dim.get_idx()
            unique_dim = unique_node_dims[dim_idx]
            converted_tiling.append((unique_dim, factor))
        return converted_tiling

    def with_modified_dimension_sizes(self, new_sizes: dict[int, int]) -> "Workload":
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
                original_tensors_dict[original_tensor.name] = original_tensor
                strides = self.strides_for_tensor(original_tensor)
                new_sizes_sorted = [new_sizes[dim] for dim in strides]
                V = np.array(list(strides.values())).T
                new_shape_t = tuple(V.dot(np.array(new_sizes_sorted)).astype(int).tolist())
                tensor_name = original_tensor.name
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
        node_map: dict[Node, Node] = {}
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
                new_inputs = tuple(cast(HasOutputs, node_map[inp]) for inp in node.inputs)
                new_output = tensor_map.get(node.output.name)
                assert new_output is not None, f"ComputationNode output tensor {node.name} must have been inferred"
                new_node = ComputationNode(
                    name=node.name,
                    inputs=new_inputs,
                    outputs=(new_output,),
                    operand_mapping=node.operand_mapping,
                )
            elif isinstance(node, TransferNode):
                new_inputs = tuple(cast(HasOutputs, node_map[inp]) for inp in node.inputs)
                new_output = tensor_map.get(node.output.name)
                assert new_output is not None, f"TransferNode output tensor {node.name} must have been inferred"
                new_node = TransferNode(
                    name=node.name,
                    inputs=new_inputs,
                    outputs=(new_output,),
                    transfer_type=node.transfer_type,
                    operand_mapping=node.operand_mapping,
                )
            elif isinstance(node, OutEdge):
                new_inputs = tuple(cast(HasOutputs, node_map[inp]) for inp in node.inputs)
                new_node = OutEdge(
                    name=node.name,
                    inputs=new_inputs,
                )
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            node_map[node] = new_node
            new_nodes.append(new_node)

        return Workload(new_nodes)

    def strides_for_tensor(self, tensor: Tensor) -> dict[LayerDim, tuple[int, ...]]:
        unique_dims, all_dims = self.unique_dimensions()
        node = next(iter(n for n in self.get_iteration_space_nodes() if tensor in n.tensors))
        mapping = node.get_mapping(tensor)
        global_mapping = self.global_mapping(node, mapping)
        one_list = np.eye(len(unique_dims), dtype=int).tolist()
        result: dict[LayerDim, tuple[int, ...]] = {}
        for unique_dim, var in zip(unique_dims, one_list, strict=True):
            var = cast(list[int], var)
            all_dims_eval = [int(dim.subs(list(zip(unique_dims, var, strict=True)))) for dim in all_dims]
            stride = global_mapping.eval(all_dims_eval, [])
            result[unique_dim] = tuple(stride)
        return result

    def visualize_to_file(self, filepath: str = "workload_graph.png"):
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
                dim_sizes = {dim: self.get_dimension_size(dim) for dim in self.get_dims(node)}
                n.set_shape("ellipse")
                n.set_label(f"{node.name}\nDims: {dim_sizes}")
                n.set_style("filled")
                n.set_fillcolor("#60bcf0")
            elif isinstance(node, TransferNode):
                dim_sizes = {dim: self.get_dimension_size(dim) for dim in self.get_dims(node)}
                n.set_shape("box")
                n.set_label(f"{node.name}\nDims: {dim_sizes}")
                n.set_style("filled")
                n.set_fillcolor("#ffcb9a")
            elif isinstance(node, InEdge):
                n.set_shape("box")
                n.set_label(f"{node.name}\nShape: {node.output.shape}")
                n.set_style("filled")
                n.set_fillcolor("#eaff76e1")
            elif isinstance(node, OutEdge):
                n.set_shape("box")
                n.set_label(f"{node.name}\nShape: {node.inputs[0].output.shape}")
                n.set_style("filled")
                n.set_fillcolor("#c2f0c2")
            else:
                raise ValueError(f"Unknown node type: {type(node)}")

        # Save to file
        dot.write_png(filepath)
        print(f"Graph saved to {filepath}")

    def get_timeslots(self) -> dict[Node, int]:
        timeslots = {}
        slot = 0
        for generation in nx.topological_generations(self):
            for node in generation:
                timeslots[node] = slot
                slot += 1
        return timeslots
