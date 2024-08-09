from typing import Any, Iterator, Literal, TypeVar, overload

import networkx as nx
from zigzag.workload.LayerNodeABC import LayerNodeABC
from zigzag.workload.ONNXWorkload import ONNXWorkload as ONNXWorkloadZigZag
from zigzag.workload.Workload import WorkloadABC as WorkloadABCZigZag

from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.node import Node

T = TypeVar("T", bound=Node)


class ONNXWorkload(ONNXWorkloadZigZag):
    """Wraps and extends the ONNXWorkload class of ZigZag"""

    def __init__(self, **attr: Any):
        """
        Collect all the algorithmic workload information here.
        :param workload: user-defined workload file (py).

        :return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)

        self.node_id_to_obj: dict[int, LayerNodeABC] = {}

    def add(self, node_id: int, node_obj: LayerNodeABC):
        """
        Add a node object to the ONNX workload graph.
        This can be a different object based on if it's an "accelerateable" node or not.
        """
        self.node_id_to_obj[node_id] = node_obj

        self.add_workload_node(node_obj)
        edges: list[tuple[LayerNodeABC, LayerNodeABC]] = []
        for op, parent_id in node_obj.input_operand_source.items():
            # for parent_id in parents:
            parent_node_obj = self.node_id_to_obj[parent_id]
            edges.append((parent_node_obj, node_obj))
            node_obj.input_operand_source[op] = parent_id  # parent_node_obj
            self.add_workload_edges_from(edges)

    def all_simple_paths(self, producer: Node, consumer: Node) -> Iterator[list[Node]]:
        """Wraps nx.all_simple_paths with type annotations. Gives all paths from producer to consumer node."""
        return nx.all_simple_paths(self, source=producer, target=consumer)  # type: ignore


class WorkloadABC(WorkloadABCZigZag[T]):
    """Wraps and extends the Workload Abstract Bass Class of ZigZag"""

    @overload
    def in_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]:
        ...  # type: ignore

    @overload
    def in_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]:
        ...  # type: ignore

    @overload
    def in_edges(self, node: T) -> list[tuple[T, T]]:
        ...  # type: ignore

    def in_edges(  # type: ignore
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        """Overwrite DiGraph method with type hints"""
        return super().in_edges(node, data)  # type: ignore

    def in_degree(self) -> Iterator[tuple[T, int]]:  # type: ignore
        return super().in_degree()  # type:ignore

    def out_degree(self) -> Iterator[tuple[T, int]]:  # type: ignore
        return super().out_degree()  # type:ignore


class ComputationNodeWorkload(WorkloadABC[ComputationNode]):
    """Workload graph with only ComputationNodes"""
