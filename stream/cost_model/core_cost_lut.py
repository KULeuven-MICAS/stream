from __future__ import annotations

import logging
import math
import os
import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import yaml

from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.core import Core
from stream.workload.node_key import node_key

if TYPE_CHECKING:
    from stream.workload.workload import ComputationNode

logger = logging.getLogger(__name__)

# Bumped whenever a ZigZag/cost-model change alters the numbers a cached entry holds; on-disk caches
# tagged with a different version are ignored (a delta-pin update travels with this bump). See plan 10.
COST_MODEL_VERSION = 1


def _to_yaml_scalar(v: Any) -> Any:
    """Coerce a value to a native scalar yaml can dump cleanly. Falls back to ``str``."""
    if v is None or isinstance(v, bool | str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return str(v)
    if f.is_integer():
        return int(f)
    return f


class CoreCostLUT:
    """Stores CoreCostEntry per (node, core) pair, with equality-aware lookups."""

    def __init__(self, cache_path: str | None = None, load: bool = True):
        self.lut: dict[ComputationNode, dict[Core, CoreCostEntry]] = {}
        # node_key -> a representative node in the LUT, for O(1) equality-aware lookup.
        self._index: dict[str, ComputationNode] = {}
        self.cache_path = cache_path
        if load and self.cache_path:
            self._maybe_load()

    def add_cost(self, node: ComputationNode, core: Core, cost: CoreCostEntry, allow_overwrite: bool = True):
        if not allow_overwrite and self.has_cost(node, core):
            raise ValueError(f"Cost entry for node {node} and core {core} already exists.")
        if node not in self.lut:
            self.lut[node] = {}
        self.lut[node][core] = cost
        self._index[node_key(node)] = node

    def has_cost(self, node: ComputationNode, core: Core) -> bool:
        return self.get_equal_node(node) is not None and node in self.lut and core in self.lut[node]

    def get_cost(self, node: ComputationNode, core: Core) -> CoreCostEntry:
        if not self.has_cost(node, core):
            raise ValueError(f"No cost entry found for node {node} and core {core}.")
        return self.lut[node][core]

    def get_nodes(self) -> list[ComputationNode]:
        return list(self.lut.keys())

    def get_cores(self, node: ComputationNode) -> list[Core]:
        return list(self.lut.get(node, {}).keys())

    def get_equal_node(self, node: ComputationNode) -> ComputationNode | None:
        return self._index.get(node_key(node))

    def get_equal_core(self, node: ComputationNode | None, core: Core) -> Core | None:
        if node is None:
            return None
        try:
            return next(c for c in self.lut[node] if c.has_same_performance(core))
        except (StopIteration, KeyError):
            return None

    def replace_node(self, old_node: ComputationNode, new_node: ComputationNode):
        # Replace the exact node when present (multiple nodes can share a key, so the key index must
        # not decide which one to pop); fall back to an equal representative only if it is not.
        target = old_node if old_node in self.lut else self.get_equal_node(old_node)
        if target is None:
            raise ValueError(f"Node {old_node} not found in LUT.")
        self.lut[new_node] = self.lut.pop(target)
        self._index.pop(node_key(target), None)
        self._index[node_key(new_node)] = new_node

    def with_nodes(self, nodes: Iterable[ComputationNode]) -> CoreCostLUT:
        """A copy that holds the entries of each same-named node under ``nodes`` instead; this LUT is unchanged."""
        view = CoreCostLUT()
        view.lut = dict(self.lut)
        view._index = dict(self._index)
        for new_node in nodes:
            view.replace_node(next(n for n in view.get_nodes() if n.name == new_node.name), new_node)
        return view

    def remove_cores_with_same_id(self, node: ComputationNode, core: Core):
        if node not in self.lut:
            return
        for c in list(self.lut[node].keys()):
            if c.id == core.id:
                self.lut[node].pop(c)

    def remove_node(self, node: ComputationNode):
        if node in self.lut:
            self.lut.pop(node)
        key = node_key(node)
        if self._index.get(key) is node:
            self._index.pop(key, None)

    def save(self):
        if not self.cache_path:
            raise ValueError("No cache_path provided.")
        with open(self.cache_path, "wb") as fp:
            pickle.dump({"version": COST_MODEL_VERSION, "lut": self.lut}, fp)
        self._save_yaml_summary()

    def _save_yaml_summary(self) -> None:
        """Write a human-readable yaml sibling next to the pickle (keys in docs/source/outputs.md)."""
        summary = {
            "nodes": [
                {
                    "node": node.name,
                    "cores": [
                        {
                            "core_id": core.id,
                            "core_type": core.core_type,
                            "latency_cycles": _to_yaml_scalar(entry.latency_total),
                            "ideal_cycles": _to_yaml_scalar(entry.ideal_cycle),
                            "ideal_temporal_cycles": _to_yaml_scalar(entry.ideal_temporal_cycle),
                            "energy_pj": _to_yaml_scalar(entry.energy_total),
                            "metadata": {str(k): _to_yaml_scalar(v) for k, v in entry.metadata.items()},
                        }
                        for core, entry in core_dict.items()
                    ],
                }
                for node, core_dict in self.lut.items()
            ]
        }
        with open(os.path.splitext(self.cache_path)[0] + ".yaml", "w") as fp:
            yaml.safe_dump(summary, fp, sort_keys=False)

    def _maybe_load(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "rb") as fp:
                data = pickle.load(fp)
            if not isinstance(data, dict) or data.get("version") != COST_MODEL_VERSION:
                raise ValueError(f"cost_model_version mismatch (need {COST_MODEL_VERSION})")
            self.lut = data["lut"]
        except Exception as e:
            logger.warning(
                "Could not load CoreCostLUT from %s (%s). Starting from empty LUT.",
                self.cache_path,
                e,
            )
            self.lut = {}
            try:
                os.remove(self.cache_path)
            except OSError:
                logger.debug("Failed to remove corrupted LUT cache at %s", self.cache_path)
        self._index = {node_key(n): n for n in self.lut}
