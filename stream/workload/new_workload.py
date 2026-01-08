from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp
from snaxc.ir.dart.affine_transform import AffineTransform
from xdsl.dialects.builtin import FixedBitwidthType, i32
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap
from zigzag.utils import DiGraphWrapper


@dataclass(frozen=True)
class Operand:
    operand_type: FixedBitwidthType


@dataclass(frozen=True)
class Node(ABC):
    name: str


@dataclass(frozen=True)
class HasOutput(Node, ABC):
    output: Operand


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

    def get_mapping(self, operand: Node | Operand) -> AffineMap:
        if operand is self.output:
            return self.operand_mapping[-1]
        for i, input in enumerate(self.inputs):
            if operand is input or operand is input.output:
                return self.operand_mapping[i]
        raise RuntimeError


class Workload(DiGraphWrapper[Node]):
    def __init__(self, nodes: Sequence[Node]):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for node in nodes:
            if isinstance(node, HasInputs):
                for input in node.inputs:
                    graph.add_edge(input, node)
        super().__init__(graph)

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

    def summary(self, dim_vals: list[int] | None = None):
        for node in self.topological_sort():
            if isinstance(node, ComputationNode):
                print(f"Node {node.name}:")
                dims = ",".join(f"d{i}" for i in self.global_idxs[node])
                print(f"Dims: {dims}")
                print(f"\t{len(node.inputs)} inputs:")
                for input, mapping in zip(node.inputs, node.operand_mapping[:-1], strict=True):
                    print(f"\t\t{input.name}: {str(mapping)}")
                    if dim_vals is not None:
                        print(f"\t\t\tdimensions: {self.global_mapping(node, mapping).eval(dim_vals, [])}")
                print("\toutput:")
                print(f"\t\toutput: {str(node.operand_mapping[-1])}")
                if dim_vals:
                    print(f"\t\t\tdimensions: {self.global_mapping(node, node.operand_mapping[-1]).eval(dim_vals, [])}")


if __name__ == "__main__":
    # Swiglu example
    input = InEdge("input", Operand(i32))
    weights_left = InEdge("weights_left", Operand(i32))
    weights_right = InEdge("weights_right", Operand(i32))
    weights_gate = InEdge("weights_gate", Operand(i32))
    gemm_left = ComputationNode(
        "gemm_left",
        (input, weights_left),
        out := Operand(i32),
        (
            AffineMap.from_callable(lambda m, n, k: (m, k)),
            AffineMap.from_callable(lambda m, n, k: (k, n)),
            AffineMap.from_callable(lambda m, n, k: (m, n)),
        ),
    )
    gemm_right = ComputationNode(
        "gemm_right",
        (input, weights_right),
        out := Operand(i32),
        (
            AffineMap.from_callable(lambda m, n, k: (m, k)),
            AffineMap.from_callable(lambda m, n, k: (k, n)),
            AffineMap.from_callable(lambda m, n, k: (m, n)),
        ),
    )
    silu = ComputationNode(
        "silu",
        (gemm_left,),
        out := Operand(i32),
        (
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
        ),
    )
    eltwise = ComputationNode(
        "eltwise",
        (silu, gemm_right),
        out := Operand(i32),
        (
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
        ),
    )
    gemm_gate = ComputationNode(
        "gemm_gate",
        (eltwise, weights_gate),
        out := Operand(i32),
        (
            AffineMap.from_callable(lambda m, n, k: (m, k)),
            AffineMap.from_callable(lambda m, n, k: (k, n)),
            AffineMap.from_callable(lambda m, n, k: (m, n)),
        ),
    )
    output = OutEdge("output", (gemm_gate,))

    nodes = (
        input,
        weights_left,
        weights_right,
        weights_gate,
        gemm_left,
        gemm_right,
        gemm_gate,
        eltwise,
        silu,
        output,
    )

    workload = Workload(nodes)

    print(f"Dimensionality of this workload is {workload.num_dims}")
    workload.summary()
    relations = AffineMap(workload.num_dims, 0, tuple(workload.dimension_relations()))
    for relation in relations.results:
        print(relation)
    transform = AffineTransform.from_affine_map(relations)
    print(transform)

    A_sp = sp.Matrix(transform.A)
    rref_A, pivots = A_sp.rref()

    n_vars = transform.A.shape[1]
    free_vars = [i for i in range(n_vars) if i not in pivots]

    print("--------------------------------------------------")
    print(f"Pivot variables: {list(pivots)}")
    print(f"Free variables (DOFs): {free_vars}")
    print(f"Number of DOFs: {len(free_vars)}")
    print("--------------------------------------------------")

    # ------------------------------------------------------------
    # Construct coordinate-aligned nullspace basis
    # ------------------------------------------------------------
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

    print("Original dimensions expressed in terms of DOFs:")
    for i, expr in enumerate(x):
        print(f"d{i:2d} = {sp.simplify(expr)}")

    print("--------------------------------------------------")

    print("Let us now fix the remaining DOFS: (hardcoded)")

    z0, z1, z2, z3 = sp.symbols("z0 z1 z2 z3")

    substitutions = [(z0, 4), (z1, 9), (z2, 3), (z3, 19)]
    dim_values = []
    for i, expr in enumerate(x):
        result = int(expr.subs(substitutions))
        dim_values.append(result)
        print(f"d{i:2d} = {result}")

    print("Workload summary:")
    workload.summary(dim_values)

    nx.draw(workload, with_labels=True, labels={node: node.name for node in workload.nodes})
    plt.show()
