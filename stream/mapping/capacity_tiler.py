"""Capacity-aware intra-core tiling for the generic auto-mapper."""

import logging
import math
from typing import Any

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode
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
    """Choose per-core intra-core tiles so every compute core's resident footprint fits its operand buffer, within
    ``fill_fraction`` of it: the headroom covers what the allocation adds beyond the tiles (handover buffers)."""

    MAX_STEADY_STATE_TILES = 1024

    def __init__(self, sub_workload: Workload, accelerator: Accelerator, fill_fraction: float = 0.9) -> None:
        self.sub_workload = sub_workload
        self.accelerator = accelerator
        self.fill_fraction = fill_fraction
        self._shapes: dict[tuple, tuple[int, ...]] = {}

    def plan(  # noqa: PLR0913
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        protected: set[LayerDim],
        seed_tiling: list[dict[str, Any]] | None = None,
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Intra-core tiling for the group, or ``[]`` when it already fits (caller keeps its own tiling).

        Dims are taken by the unique dim they step along fastest, so a strided reader's ``s*d + r`` is tiled as ``d``,
        in whole strides. What a memory holds is :meth:`memory_bits`."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        protected = {lead(d)[0] for d in protected}
        budgets = {m: int(cap * self.fill_fraction) for m, cap in self._capacities(cns, cores_per_node).items()}

        # Per candidate dim: per-core size (full / inter-core unroll) and divisor tile sizes.
        per_core: dict[LayerDim, int] = {}
        divisors: dict[LayerDim, list[int]] = {}
        for dim in dict.fromkeys(lead(d)[0] for cn in cns for d in self.sub_workload.get_dims(cn)):
            full = self.sub_workload.get_dimension_size(dim)
            u = unroll.get(dim, 1)
            size = full // u if u > 1 and full % u == 0 else full
            if dim not in protected and size > 1:
                per_core[dim] = size
                divisors[dim] = _divisors_desc(size)
        seeded = self._seed_resident(cns, seed_tiling or [], per_core)
        resident: dict[LayerDim, int] = {dim: seeded.get(dim, per_core[dim]) for dim in per_core}

        def overflow() -> float:
            """Largest amount by which any memory exceeds its budget (<= 0 means the whole group fits)."""
            held = self.memory_bits(cns, cores_per_node, unroll, resident, node_unroll)
            return max((bits - budgets[m] for m, bits in held.items() if budgets[m] > 0), default=0.0)

        if overflow() <= 0 or not per_core:
            return []

        # Greedy: repeatedly shrink the dim that most reduces the worst overflow, keeping others large. Dims every node
        # keeps as an output axis are tried alone first, so a contraction is streamed only when they cannot make it fit.
        reductions = {
            lead(dims[p])[0]
            for cn in cns
            for dims in [self.sub_workload.get_dims(cn)]
            for p, kind in derive_iterator_types(cn).items()
            if kind != IteratorType.PARALLEL and p < len(dims)
        }
        start = dict(resident)
        for candidates in ([d for d in per_core if d not in reductions], list(per_core)):
            resident.update(start)
            while overflow() > 0 and (step := self._shrink(candidates, resident, per_core, divisors, overflow)):
                resident[step[0]] = step[1]
            if overflow() <= 0:
                return self._emit(cns, resident, per_core)
        return []

    def _shrink(self, candidates, resident, per_core, divisors, overflow) -> tuple[LayerDim, int] | None:
        """The dim and smaller tile that save the most overflow per doubling of the steady-state tiles, within the tile
        budget; None when no step saves any. Looking past the next tile skips one that only starts double buffering
        what it shrinks."""
        base = overflow()
        best, best_rate = None, 0.0
        for dim in candidates:
            keep = resident[dim]
            for smaller in (d for d in divisors[dim] if d < keep):
                resident[dim] = smaller
                if math.prod(per_core[d] / resident[d] for d in per_core) > self.MAX_STEADY_STATE_TILES:
                    break
                if (rate := (base - overflow()) / math.log2(keep / smaller)) > best_rate:
                    best, best_rate = (dim, smaller), rate
            resident[dim] = keep
        return best

    def memory_bits(
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None = None,
    ) -> dict[int, float]:
        """Bits each memory holds when every unique dim of ``resident`` spans that tile on a core: per node a core
        runs, the tile of each operand the node touches under its own split (``node_unroll``, else the group's
        ``unroll`` on a multi-core node), windows included, each distinct tile of a core once and twice where a
        temporal loop steps it (double buffered); cores sharing a memory fill it together."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        unique, _ = self.sub_workload.unique_dimensions()
        full = {z: self.sub_workload.get_dimension_size(z) for z in unique}
        held: dict[int, dict[tuple[str, tuple[int, ...]], float]] = {}  # per core, each distinct tile once
        for cn in cns:
            cores = cores_per_node.get(cn, [])
            if node_unroll is not None:
                own = {lead(d)[0]: f for d, f in node_unroll.get(cn, {}).items()}
            else:
                own = {d: f for d, f in unroll.items() if d in self._node_dims(cn)} if len(cores) > 1 else {}
            sizes = full | {d: min(full[d], r * unroll.get(d, 1) // own.get(d, 1)) for d, r in resident.items()}
            stepped = {d for d, r in resident.items() if r * unroll.get(d, 1) < full[d]}
            dims = self.sub_workload.get_dims(cn)
            for t in dict.fromkeys(cn.tensors):
                key = (t.name, cn.name, tuple(sizes[z] for z in unique))
                if (shape := self._shapes.get(key)) is None:
                    shape = self._shapes[key] = self.sub_workload.get_tensor_shape_with_dimension_sizes(t, sizes, cn)
                buffers = 2 if any(lead(dims[p])[0] in stepped for p in map_dim_positions(cn.get_mapping(t))) else 1
                for core in cores:
                    held.setdefault(core.id, {})[(t.name, shape)] = buffers * math.prod(shape) * t.operand_type.bitwidth
        memories: dict[int, float] = {}
        for core_id, tiles in held.items():
            memory = self.accelerator.memory_of(self.accelerator.get_core(core_id)).id
            memories[memory] = memories.get(memory, 0.0) + sum(tiles.values())
        return memories

    def _capacities(
        self, cns: tuple[ComputationNode, ...], cores_per_node: dict[ComputationNode, list[Core]]
    ) -> dict[int, int]:
        """Each memory the group's cores use, with its capacity."""
        return {
            self.accelerator.memory_of(core).id: core.get_memory_capacity()
            for cn in cns
            for core in cores_per_node.get(cn, [])
        }

    def _node_dims(self, cn: ComputationNode) -> set[LayerDim]:
        """The unique dims a node steps along."""
        return {self.sub_workload.leading_dim(d)[0] for d in self.sub_workload.get_dims(cn)}

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
            if idx < len(dims) and (dim := self.sub_workload.leading_dim(dims[idx])[0]) in per_core:
                seeded[dim] = int(entry["tile"])
        return seeded

    def _emit(
        self,
        cns: tuple[ComputationNode, ...],
        resident: dict[LayerDim, int],
        per_core: dict[LayerDim, int],
    ) -> list[dict[str, Any]]:
        """One entry per tiled dim (resident tile < the per-core slice), naming a node that has the dim, else a node dim
        stepping along it, its tile scaled by the stride."""
        entries: list[dict[str, Any]] = []
        for dim, tile in resident.items():
            if tile >= per_core.get(dim, tile):
                continue
            steps = [
                (abs(coefficient), cn.name, idx)
                for cn in cns
                for idx, d in enumerate(self.sub_workload.get_dims(cn))
                for lead, coefficient in [self.sub_workload.leading_dim(d)]
                if lead == dim
            ]
            if steps:
                stride, name, idx = min(steps, key=lambda step: step[0])
                entries.append({"dim": f"{name}.D{idx}", "tile": tile * stride})
        return entries
