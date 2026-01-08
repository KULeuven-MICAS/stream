from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import networkx as nx
from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper


@dataclass(frozen=True)
class Tensor:
    operand_type: FixedBitwidthType
    shape: tuple[int, ...]


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

    def get_dimension_size(self, layer_dim: str) -> int:
        dim_index = int(layer_dim.strip("D"))
        return self.output.shape[dim_index]  # TODO: Probably not always of output tensor


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
        Determine unique global indeces for each iteration dimension in this workload
        """
        global_dimension_idxs: dict[Node, range] = {}
        idx = 0
        for node in nx.lexicographical_topological_sort(self, key=lambda node: node.name):
            if isinstance(node, ComputationNode):
                global_dimension_idxs[node] = range(idx, idx + node.num_dims)
                idx += node.num_dims
        return global_dimension_idxs

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
