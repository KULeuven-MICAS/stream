from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import networkx as nx
import sympy as sp
from snaxc.ir.dart.affine_transform import AffineTransform
from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper

from stream.datatypes import LayerDim


@dataclass(frozen=True)
class Tensor:
    operand_type: FixedBitwidthType
    shape: tuple[int, ...]

    def size_elements(self) -> int:
        return sp.prod(self.shape)

    def size_bits(self) -> int:
        return self.operand_type.bitwidth * self.size_elements()


@dataclass(frozen=True)
class Node(ABC):
    name: str


@dataclass(frozen=True)
class HasOutput(Node, ABC):
    output: Tensor


@dataclass(frozen=True)
class HasInputs(Node, ABC):
    inputs: tuple[HasOutput, ...]


@dataclass(frozen=True)
class InEdge(HasOutput): ...


@dataclass(frozen=True)
class OutEdge(HasInputs): ...


@dataclass(frozen=True)
class ComputationNode(HasOutput, HasInputs):
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


class Workload(DiGraphWrapper[Node]):
    def __init__(self, nodes: Sequence[Node]):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for node in nodes:
            if isinstance(node, HasInputs):
                for input in node.inputs:
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
        return sum(node.num_dims for node in self.nodes if isinstance(node, ComputationNode))

    @property
    def global_idxs(self):
        """
        Determine unique global indices for each dimension in this workload
        """
        global_dimension_idxs: dict[Node, range] = {}
        idx = 0
        for node in nx.lexicographical_topological_sort(self, key=lambda node: node.name):
            if isinstance(node, ComputationNode):
                global_dimension_idxs[node] = range(idx, idx + node.num_dims)
                idx += node.num_dims
        return global_dimension_idxs

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        seen = set()
        tensors = []
        for node in self.get_computation_nodes():
            for tensor in node.tensors:
                if tensor.operand_type not in seen:
                    seen.add(tensor.operand_type)
                    tensors.append(tensor)
        return tuple(tensors)

    def global_mapping(self, node: ComputationNode, mapping: AffineMap):
        return mapping.replace_dims_and_symbols(
            [AffineDimExpr(i) for i in self.global_idxs[node]], [], self.num_dims, 0
        )

    def dimension_relations(self) -> Sequence[AffineExpr]:
        result = []
        # Relations between shared intermediate tensors:
        for edge in self.edges:
            if isinstance(edge[0], ComputationNode) and isinstance(edge[1], ComputationNode):
                mapping_out = self.global_mapping(edge[0], edge[0].get_mapping(edge[0].output))
                mapping_in = self.global_mapping(edge[1], edge[1].get_mapping(edge[0]))
                for expr_out, expr_in in zip(mapping_out.results, mapping_in.results, strict=True):
                    # expr_out == expr_in <=> expr_out - expr_in == 0
                    result.append(expr_out - expr_in)
        # Relations between shared inputs:
        for node in self.nodes:
            if isinstance(node, InEdge):
                all_users = [cast(ComputationNode, out) for (_, out) in self.out_edges(node)]
                for a, b in combinations(all_users, 2):
                    mapping_a = self.global_mapping(a, a.get_mapping(node))
                    mapping_b = self.global_mapping(b, b.get_mapping(node))
                    for expr_a, expr_b in zip(mapping_a.results, mapping_b.results, strict=True):
                        result.append(expr_a - expr_b)
        return result

    def get_computation_nodes(self) -> tuple[ComputationNode, ...]:
        return tuple(cast(ComputationNode, node) for node in self.nodes if isinstance(node, ComputationNode))

    def get_in_edges(self) -> tuple[InEdge, ...]:
        return tuple(cast(InEdge, node) for node in self.nodes if isinstance(node, InEdge))

    def get_out_edges(self) -> tuple[OutEdge, ...]:
        return tuple(cast(OutEdge, node) for node in self.nodes if isinstance(node, OutEdge))

    def get_dimension_sizes(self) -> tuple[int, ...]:
        results = []
        shapes: list[int] = []
        for node in self.get_computation_nodes():
            operands = [inp.output for inp in node.inputs] + [node.output]
            for operand, mapping in zip(operands, node.operand_mapping, strict=True):
                global_mapping = self.global_mapping(node, mapping)
                results.extend(global_mapping.results)
                shapes.extend(operand.shape)
        total_map = AffineMap(self.num_dims, 0, tuple(results))
        return total_map.inverse_permutation().eval(shapes, [])

    def get_dims(self, node: ComputationNode) -> list[LayerDim]:
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

    def with_modified_dimension_sizes(self, new_sizes: dict[int, int]) -> "Workload":
        """Create a new workload where the dimension sizes of the given global dimension indices are modified to the new
        sizes provided in new_sizes.

        This recreates all tensors (and nodes referencing them) so tensor shapes stay consistent with the updated
        global loop sizes.
        """
        # Start from the current global loop ranges and apply overrides.
        _, all_dims = self.unique_dimensions()
        new_sizes_for_all_dims = [new_sizes[dim] for dim in all_dims]

        # Infer the updated shape for every tensor based on how each operand maps onto global dims.
        inferred_shapes: dict[str, tuple[int, ...]] = {}
        original_tensors_dict: dict[str, Tensor] = {}
        for node in self.get_computation_nodes():
            original_tensors = node.tensors
            tensor_names = [inp.name for inp in node.inputs] + [node.name]  # output name is node name
            for original_tensor, tensor_name, mapping in zip(
                original_tensors, tensor_names, node.operand_mapping, strict=True
            ):
                original_tensors_dict[tensor_name] = original_tensor
                global_mapping = self.global_mapping(node, mapping)
                new_shape: list[int] = []
                for expr in global_mapping.results:
                    if isinstance(expr, AffineDimExpr):
                        new_shape.append(new_sizes_for_all_dims[expr.position])
                    else:
                        raise NotImplementedError(
                            "Updating tensor shapes only supports dimension projections/permutations "
                            f"(AffineDimExpr results); got {expr} in mapping for node '{node.name}'."
                        )
                new_shape_t = tuple(new_shape)
                if tensor_name in inferred_shapes and inferred_shapes[tensor_name] != new_shape_t:
                    raise ValueError(
                        "Inconsistent inferred shapes for a shared tensor; "
                        f"tensor={tensor_name}, existing={inferred_shapes[tensor_name]}, new={new_shape_t}."
                    )
                inferred_shapes[tensor_name] = new_shape_t

        # Create new Tensor objects with the inferred shapes.
        tensor_map: dict[str, Tensor] = {}
        for tensor_name, new_shape_t in inferred_shapes.items():
            original_tensor = original_tensors_dict[tensor_name]
            new_output = Tensor(
                operand_type=original_tensor.operand_type,
                shape=new_shape_t,
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
                    output=new_output,
                )
            elif isinstance(node, ComputationNode):
                new_inputs = tuple(cast(HasOutput, node_map[inp]) for inp in node.inputs)
                new_output = tensor_map.get(node.name)
                assert new_output is not None, f"ComputationNode output tensor {node.name} must have been inferred"
                new_node = ComputationNode(
                    name=node.name,
                    inputs=new_inputs,
                    output=new_output,
                    operand_mapping=node.operand_mapping,
                )
            elif isinstance(node, OutEdge):
                new_inputs = tuple(cast(HasOutput, node_map[inp]) for inp in node.inputs)
                new_node = OutEdge(
                    name=node.name,
                    inputs=new_inputs,
                )
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

            node_map[node] = new_node
            new_nodes.append(new_node)

        return Workload(new_nodes)
