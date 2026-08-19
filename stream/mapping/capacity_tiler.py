"""Capacity-aware intra-core tiling for the generic auto-mapper."""

import logging
import math
from typing import Any

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.workload.affine_access import map_dim_positions
from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


def _divisors_desc(n: int) -> list[int]:
    """Divisors of ``n``, largest first."""
    if n <= 1:
        return [1]
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return sorted(small + large, reverse=True)


class CapacityTiler:
    """Choose per-core intra-core tiles so every compute core's resident footprint fits its operand buffer."""

    MAX_STEADY_STATE_TILES = 1024

    def __init__(self, sub_workload: Workload, accelerator: Accelerator, fill_fraction: float = 0.5) -> None:
        self.sub_workload = sub_workload
        self.accelerator = accelerator
        self.fill_fraction = fill_fraction

    def plan(  # noqa: PLR0912, PLR0915
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        protected: set[LayerDim],
        seed_tiling: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Intra-core tiling for the group, or ``[]`` when it already fits (caller keeps its own tiling)."""
        # Per physical core: its capacity and the distinct tensors resident on it (deduped by name).
        core_cap: dict[int, int] = {}
        core_tensors: dict[int, list[tuple[Tensor, frozenset[LayerDim]]]] = {}
        core_seen: dict[int, set[str]] = {}
        for cn in cns:
            node_tensors = self._node_tensors(cn)
            for core in cores_per_node.get(cn, []):
                core_cap[core.id] = core.get_memory_capacity()
                bucket = core_tensors.setdefault(core.id, [])
                seen = core_seen.setdefault(core.id, set())
                for tensor, dims in node_tensors:
                    if tensor.name not in seen:
                        seen.add(tensor.name)
                        bucket.append((tensor, dims))
        budgets = {cid: int(cap * self.fill_fraction) for cid, cap in core_cap.items()}

        # Per candidate dim: per-core size (full / inter-core unroll) and divisor tile sizes.
        all_dims: set[LayerDim] = {d for ts in core_tensors.values() for entry in ts for d in entry[1]}
        per_core: dict[LayerDim, int] = {}
        divisors: dict[LayerDim, list[int]] = {}
        for dim in all_dims - protected:
            full = self.sub_workload.get_dimension_size(dim)
            u = unroll.get(dim, 1)
            size = full // u if u > 1 and full % u == 0 else full
            if size > 1:
                per_core[dim] = size
                divisors[dim] = _divisors_desc(size)

        seeded = self._seed_resident(cns, seed_tiling or [], per_core)
        resident: dict[LayerDim, int] = {dim: seeded.get(dim, per_core[dim]) for dim in per_core}

        # Per core, each resident tensor as (base bits, scaling dims) so overflow() is cheap arithmetic.
        base_of: dict[str, tuple[float, tuple[LayerDim, ...]]] = {}
        core_terms: dict[int, list[tuple[float, tuple[LayerDim, ...]]]] = {}
        for cid, tensors in core_tensors.items():
            terms: list[tuple[float, tuple[LayerDim, ...]]] = []
            for t, dims in tensors:
                cached = base_of.get(t.name)
                if cached is None:
                    full_bits = math.prod(t.shape) * t.operand_type.bitwidth
                    split = math.prod(unroll[d] for d in dims if unroll.get(d, 1) > 1)
                    cached = (full_bits / split, tuple(d for d in dims if d in per_core))
                    base_of[t.name] = cached
                terms.append(cached)
            core_terms[cid] = terms

        def overflow() -> float:
            """Largest amount by which any core exceeds its budget (<= 0 means the whole group fits)."""
            return max(
                (
                    sum(
                        base * math.prod(resident[d] / per_core[d] for d in idx) if idx else base
                        for base, idx in core_terms[cid]
                    )
                    - budgets[cid]
                    for cid in core_cap
                    if budgets[cid] > 0
                ),
                default=0.0,
            )

        if overflow() <= 0 or not per_core:
            return []

        # Greedy: repeatedly shrink the dim that most reduces the worst overflow, keeping others large.
        def ss_tiles() -> float:
            return math.prod(per_core[d] / resident[d] for d in per_core)

        while overflow() > 0:
            best_dim: LayerDim | None = None
            best_after = overflow()
            for dim in per_core:
                smaller = next((d for d in divisors[dim] if d < resident[dim]), None)
                if smaller is None:
                    continue
                keep = resident[dim]
                resident[dim] = smaller
                after, tiles = overflow(), ss_tiles()
                resident[dim] = keep
                if after < best_after and tiles <= self.MAX_STEADY_STATE_TILES:
                    best_after, best_dim = after, dim
            if best_dim is None:
                break
            resident[best_dim] = next(d for d in divisors[best_dim] if d < resident[best_dim])

        if overflow() > 0:
            return []

        return self._emit(cns, resident, per_core)

    def _node_tensors(self, cn: ComputationNode) -> list[tuple[Tensor, frozenset[LayerDim]]]:
        """One node's own operands, each with the global dims that index it."""
        node_dims = self.sub_workload.get_dims(cn)
        seen: set[str] = set()
        out: list[tuple[Tensor, frozenset[LayerDim]]] = []
        for tensor in cn.tensors:
            if tensor.name in seen:
                continue
            seen.add(tensor.name)
            dims = frozenset(node_dims[p] for p in map_dim_positions(cn.get_mapping(tensor)) if p < len(node_dims))
            out.append((tensor, dims))
        return out

    def _seed_resident(
        self,
        cns: tuple[ComputationNode, ...],
        seed_tiling: list[dict[str, Any]],
        per_core: dict[LayerDim, int],
    ) -> dict[LayerDim, int]:
        """Resolve seeded ``{"dim": "Node.Dn", "tile": T}`` entries to ``{global dim: T}`` for dims this group tiles."""
        by_name = {cn.name: cn for cn in cns}
        seeded: dict[LayerDim, int] = {}
        for entry in seed_tiling:
            node_name, _, pos = str(entry["dim"]).partition(".D")
            cn = by_name.get(node_name)
            if cn is None or not pos.isdigit():
                continue
            dims = self.sub_workload.get_dims(cn)
            idx = int(pos)
            if idx < len(dims) and dims[idx] in per_core:
                seeded[dims[idx]] = int(entry["tile"])
        return seeded

    def _emit(
        self,
        cns: tuple[ComputationNode, ...],
        resident: dict[LayerDim, int],
        per_core: dict[LayerDim, int],
    ) -> list[dict[str, Any]]:
        """One entry per tiled dim (resident tile < the per-core slice), naming a node that has the dim."""
        entries: list[dict[str, Any]] = []
        for dim, tile in resident.items():
            if tile >= per_core.get(dim, tile):
                continue
            for cn in cns:
                dims = self.sub_workload.get_dims(cn)
                if dim in dims:
                    entries.append({"dim": f"{cn.name}.D{dims.index(dim)}", "tile": tile})
                    break
        return entries
