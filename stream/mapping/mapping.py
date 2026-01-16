from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.datatypes import InterCoreTiling
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.workload import Node, Workload

Resource = Core | tuple[CommunicationLink, ...]


@dataclass(slots=True)
class NodeMapping:
    """Mapping attributes for a single workload Node. A Node can be either a ComputationNode or a TransferNode."""

    resource_allocation: tuple[Resource, ...] = field(default_factory=tuple)
    inter_core_tiling: InterCoreTiling = field(default_factory=tuple)
    memory_allocation: tuple[Core, ...] = field(default_factory=tuple)
    kernel: AIEKernel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_allocation, tuple):
            raise TypeError("core_allocation must be a tuple of Resource objects")

        if not isinstance(self.inter_core_tiling, tuple):
            raise TypeError("inter_core_tiling must be a tuple of (dimension, factor) tuples")

        for item in self.inter_core_tiling:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError("inter_core_tiling entries must be tuples of length 2: (dim, factor)")
            dim, factor = item
            if not isinstance(factor, int) or factor <= 0:
                raise ValueError(f"Tiling factor for '{dim}' must be a positive int, got {factor!r}.")

        if self.kernel is not None:
            if not isinstance(self.kernel, AIEKernel):
                raise TypeError("Kernel must be an AIEKernel instance. Other kernel types are not supported yet.")


class Mapping:
    """Holds per-Node mapping information across multiple layers."""

    def __init__(self, initial: dict[Node, NodeMapping] | None = None) -> None:
        self._by_node: dict[Node, NodeMapping] = {}
        if initial:
            for node, layer_mapping in initial.items():
                self.set(node, layer_mapping)

    def set(self, node: Node, layer_mapping: NodeMapping) -> None:
        if node is None:
            raise ValueError("node must not be None")
        if not isinstance(layer_mapping, NodeMapping):
            raise TypeError("layer_mapping must be a LayerMapping instance")
        self._by_node[node] = layer_mapping

    def set_for_node(
        self,
        node: Node,
        resource_allocation: tuple[Resource, ...],
        inter_core_tiling: InterCoreTiling,
        memory_allocation: tuple[Core, ...] = (),
        kernel: AIEKernel | None = None,
    ) -> None:
        self.set(
            node,
            NodeMapping(
                resource_allocation=resource_allocation,
                inter_core_tiling=inter_core_tiling,
                kernel=kernel,
                memory_allocation=memory_allocation,
            ),
        )

    def get(self, node: Node) -> NodeMapping:
        layer_mapping = self._by_node.get(node)
        assert layer_mapping is not None, f"No LayerMapping found for node {node}"
        return layer_mapping

    def remove(self, node: Node) -> None:
        """Remove a node entry from the mapping."""
        if node not in self._by_node:
            raise KeyError(f"Node {node} not found in mapping")
        del self._by_node[node]

    def __getitem__(self, node: Node) -> NodeMapping:
        return self._by_node[node]

    def __setitem__(self, node: Node, layer_mapping: NodeMapping) -> None:
        self.set(node, layer_mapping)

    def __contains__(self, node: object) -> bool:
        return node in self._by_node

    def __len__(self) -> int:
        return len(self._by_node)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._by_node)

    def items(self) -> Iterable[tuple[Node, NodeMapping]]:
        return self._by_node.items()

    def nodes(self) -> Iterable[Node]:
        return self._by_node.keys()

    def values(self) -> Iterable[NodeMapping]:
        return self._by_node.values()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """
        Serialize to a JSON-friendly dict keyed by a stable node identifier.

        Uses node.name if present, else repr(node).
        """
        out: dict[str, dict[str, Any]] = {}
        for node, lm in self._by_node.items():
            node_key = getattr(node, "name", None) or repr(node)
            out[str(node_key)] = {
                "resource_allocation": [repr(c) for c in lm.resource_allocation],
                "inter_core_tiling": list(lm.inter_core_tiling),
                "kernel": dict(lm.kernel) if lm.kernel is not None else None,
            }
        return out

    def with_updated_workload(self, workload: Workload) -> Mapping:
        """Return a new Mapping instance with the same LayerMappings but updated to a new Workload.
        This is useful when the Workload has been modified (e.g., tiled) and we want to keep the same
        mapping information.
        This assumes the names of the nodes are unique identifiers that remain unchanged between workloads.

        Args:
            workload (Workload): The new Workload instance.

        Returns:
            Mapping: A new Mapping instance with updated workload reference.
        """
        new_mapping = Mapping()
        for node in self.nodes():
            updated_node = workload.get_node_by_name(node.name)
            new_mapping.set(updated_node, self.get(node))
        return new_mapping
