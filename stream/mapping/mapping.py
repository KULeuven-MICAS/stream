from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.datatypes import InterCoreTiling
from stream.hardware.architecture.core import Core
from stream.workload.workload import Node


@dataclass(slots=True)
class LayerMapping:
    """Mapping attributes for a single workload Node (layer)."""

    core_allocation: list[Core] = field(default_factory=list)
    inter_core_tiling: InterCoreTiling = field(default_factory=list)
    kernel: AIEKernel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.core_allocation, list):
            raise TypeError("core_allocation must be a list of Core objects")

        if not isinstance(self.inter_core_tiling, list):
            raise TypeError("inter_core_tiling must be a list of (dimension, factor) tuples")

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

    def __init__(self, initial: dict[Node, LayerMapping] | None = None) -> None:
        self._by_node: dict[Node, LayerMapping] = {}
        if initial:
            for node, layer_mapping in initial.items():
                self.set(node, layer_mapping)

    def set(self, node: Node, layer_mapping: LayerMapping) -> None:
        if node is None:
            raise ValueError("node must not be None")
        if not isinstance(layer_mapping, LayerMapping):
            raise TypeError("layer_mapping must be a LayerMapping instance")
        self._by_node[node] = layer_mapping

    def set_for_node(
        self,
        node: Node,
        core_allocation: list[Core],
        inter_core_tiling: InterCoreTiling,
        kernel: AIEKernel | None = None,
    ) -> None:
        self.set(
            node,
            LayerMapping(
                core_allocation=core_allocation,
                inter_core_tiling=inter_core_tiling,
                kernel=kernel,
            ),
        )

    def get(self, node: Node, default: LayerMapping | None = None) -> LayerMapping | None:
        return self._by_node.get(node, default)

    def __getitem__(self, node: Node) -> LayerMapping:
        return self._by_node[node]

    def __setitem__(self, node: Node, layer_mapping: LayerMapping) -> None:
        self.set(node, layer_mapping)

    def __contains__(self, node: object) -> bool:
        return node in self._by_node

    def __len__(self) -> int:
        return len(self._by_node)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._by_node)

    def items(self) -> Iterable[tuple[Node, LayerMapping]]:
        return self._by_node.items()

    def nodes(self) -> Iterable[Node]:
        return self._by_node.keys()

    def values(self) -> Iterable[LayerMapping]:
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
                "core_allocation": [repr(c) for c in lm.core_allocation],
                "inter_core_tiling": list(lm.inter_core_tiling),
                "kernel": dict(lm.kernel) if lm.kernel is not None else None,
            }
        return out
