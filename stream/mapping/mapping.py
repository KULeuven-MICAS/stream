from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.hardware.architecture.core import Core
from stream.workload.workload import Node

InterCoreTiling = List[Tuple[str, int]]


def _is_valid_dim(dim: str) -> bool:
    # Expected: "Dx" where x is an integer index (e.g., "D0", "D1", ...)
    return isinstance(dim, str) and len(dim) >= 2 and dim[0] == "D" and dim[1:].isdigit()


@dataclass(slots=True)
class LayerMapping:
    """Mapping attributes for a single workload Node (layer)."""

    core_allocation: List["Core"] = field(default_factory=list)
    inter_core_tiling: InterCoreTiling = field(default_factory=list)
    kernel: Optional[AIEKernel] = None

    def __post_init__(self) -> None:
        if not isinstance(self.core_allocation, list):
            raise TypeError("core_allocation must be a list of Core objects")

        if not isinstance(self.inter_core_tiling, list):
            raise TypeError("inter_core_tiling must be a list of (dimension, factor) tuples")

        for item in self.inter_core_tiling:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError("inter_core_tiling entries must be tuples of length 2: (dim, factor)")
            dim, factor = item
            if not _is_valid_dim(dim):
                raise ValueError(f"Invalid dimension '{dim}'. Expected format 'Dx' (e.g., 'D0').")
            if not isinstance(factor, int) or factor <= 0:
                raise ValueError(f"Tiling factor for '{dim}' must be a positive int, got {factor!r}.")

        if self.kernel is not None:
            if not isinstance(self.kernel, AIEKernel):
                raise TypeError("Kernel must be an AIEKernel instance. Other kernel types are not supported yet.")


class Mapping:
    """Holds per-Node mapping information across multiple layers."""

    def __init__(self, initial: Optional[dict["Node", LayerMapping]] = None) -> None:
        self._by_node: Dict["Node", LayerMapping] = {}
        if initial:
            for node, layer_mapping in initial.items():
                self.set(node, layer_mapping)

    def set(self, node: "Node", layer_mapping: LayerMapping) -> None:
        if node is None:
            raise ValueError("node must not be None")
        if not isinstance(layer_mapping, LayerMapping):
            raise TypeError("layer_mapping must be a LayerMapping instance")
        self._by_node[node] = layer_mapping

    def set_for_node(
        self,
        node: "Node",
        core_allocation: List["Core"],
        inter_core_tiling: InterCoreTiling,
        kernel: Optional[AIEKernel] = None,
    ) -> None:
        self.set(
            node,
            LayerMapping(
                core_allocation=core_allocation,
                inter_core_tiling=inter_core_tiling,
                kernel=kernel,
            ),
        )

    def get(self, node: "Node", default: Optional[LayerMapping] = None) -> Optional[LayerMapping]:
        return self._by_node.get(node, default)

    def __getitem__(self, node: "Node") -> LayerMapping:
        return self._by_node[node]

    def __setitem__(self, node: "Node", layer_mapping: LayerMapping) -> None:
        self.set(node, layer_mapping)

    def __contains__(self, node: object) -> bool:
        return node in self._by_node

    def __len__(self) -> int:
        return len(self._by_node)

    def __iter__(self) -> Iterator["Node"]:
        return iter(self._by_node)

    def items(self) -> Iterable[tuple["Node", LayerMapping]]:
        return self._by_node.items()

    def nodes(self) -> Iterable["Node"]:
        return self._by_node.keys()

    def values(self) -> Iterable[LayerMapping]:
        return self._by_node.values()

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Serialize to a JSON-friendly dict keyed by a stable node identifier.

        Uses node.name if present, else repr(node).
        """
        out: Dict[str, Dict[str, Any]] = {}
        for node, lm in self._by_node.items():
            node_key = getattr(node, "name", None) or repr(node)
            out[str(node_key)] = {
                "core_allocation": [repr(c) for c in lm.core_allocation],
                "inter_core_tiling": list(lm.inter_core_tiling),
                "kernel": dict(lm.kernel) if lm.kernel is not None else None,
            }
        return out