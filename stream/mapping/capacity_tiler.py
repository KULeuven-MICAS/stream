"""Capacity-aware intra-core tiling for the generic auto-mapper."""

import logging
import math
from typing import Any

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import role_of
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
    ``fill_fraction`` of it: the headroom covers what the allocation adds beyond the tiles (handover buffers). Of the
    tiles that fit, those that keep the most of each node's array filled are taken, then cut finer, as long as that
    keeps the arrays as full, into at least ``PIPELINE_TILES``."""

    MAX_STEADY_STATE_TILES = 1024
    PIPELINE_TILES = 8
    """Steady-state tiles a group is cut into where its arrays stay as full, so that each tile's transfers overlap the
    others' compute rather than the whole group's loading, computing and storing following each other."""

    def __init__(self, sub_workload: Workload, accelerator: Accelerator, fill_fraction: float = 0.9) -> None:
        self.sub_workload = sub_workload
        self.accelerator = accelerator
        self.fill_fraction = fill_fraction
        self._shapes: dict[tuple, tuple[int, ...]] = {}
        self._unrolling: dict[tuple[str, str], dict[LayerDim, int]] = {}

    def plan(  # noqa: PLR0913
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        protected: set[LayerDim],
        seed_tiling: list[dict[str, Any]] | None = None,
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Intra-core tiling for the group, ``[]`` when it already fits (caller keeps its own tiling), or None when no
        tiling within :attr:`MAX_STEADY_STATE_TILES` fits.

        Dims are taken by the unique dim they step along fastest, so a strided reader's ``s*d + r`` is tiled as ``d``,
        in whole strides. What a memory holds is :meth:`memory_bits`."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        protected = {lead(d)[0] for d in protected}
        budgets = {m: int(cap * self.fill_fraction) for m, cap in self._capacities(cns, cores_per_node).items()}

        # Per candidate dim: per-core size (full / inter-core unroll) and divisor tile sizes.
        per_core: dict[LayerDim, int] = {}
        divisors: dict[LayerDim, list[int]] = {}
        slices = self._slices(unroll)
        for dim in dict.fromkeys(lead(d)[0] for cn in cns for d in self.sub_workload.get_dims(cn)):
            if dim not in protected and (size := slices[dim]) > 1:
                per_core[dim] = size
                divisors[dim] = _divisors_desc(size)
        seeded = self._seed_resident(cns, seed_tiling or [], per_core)
        resident: dict[LayerDim, int] = {dim: seeded.get(dim, per_core[dim]) for dim in per_core}

        def overflow() -> float:
            """Largest amount by which any memory exceeds its budget (0 means the whole group fits). Never negative, so
            a shrink gets no credit for going past fitting."""
            held = self.memory_bits(cns, cores_per_node, unroll, resident, node_unroll)
            return max([0.0, *(bits - budgets[m] for m, bits in held.items() if budgets[m] > 0)])

        def fill() -> float:
            return self.array_fill(cns, cores_per_node, unroll, resident, node_unroll)

        aligned = self._aligned(cns, cores_per_node, unroll, resident, node_unroll, per_core)

        if not per_core:
            return []
        seed = dict(resident)

        # Greedy: repeatedly shrink the dim that most reduces the worst overflow, keeping others large. Dims every node
        # keeps as an output axis are tried alone first, so a contraction is streamed only when they cannot make it fit.
        # Shrinking starts both from the seed and from whole per-core slices, since a seed that tiles one dim small can
        # leave no room to keep the arrays filled along the others.
        reduced_by = {
            cn: {
                lead(dims[p])[0]
                for p, kind in derive_iterator_types(cn).items()
                if kind != IteratorType.PARALLEL and p < len(dims)
            }
            for cn in cns
            for dims in [self.sub_workload.get_dims(cn)]
        }
        reductions = set().union(*reduced_by.values())
        # A tiling that keeps the extent each array unrolls a dim over (or the whole slice where it is smaller) wins
        # where it cuts no contraction the best other tiling keeps whole.
        floors = {
            d: min(f, per_core[d]) for d, f in self._unrolled_extents(cns, cores_per_node).items() if d in per_core
        }
        floored = {d: [t for t in ts if t >= floors.get(d, 1)] for d, ts in divisors.items()}
        best_fill = self._best_fill(cns, cores_per_node, unroll, per_core, node_unroll)
        found = [
            self._fits(
                seed,
                per_core,
                allowed,
                reductions,
                resident=resident,
                overflow=overflow,
                fill=fill,
                aligned=aligned,
                best_fill=best_fill,
                seeded=seeded,
            )
            for allowed, seeded in (
                (floored, all(seed[d] >= floors.get(d, 1) for d in per_core)),
                (divisors, True),
            )
        ]
        if not found[1]:
            return None

        def contractions_cut(tiles: dict[LayerDim, int]) -> set[LayerDim]:
            return {d for d in reductions & per_core.keys() if tiles[d] < per_core[d]}

        other = contractions_cut(max(found[1], key=lambda fit: fit[0])[1])
        kept = [fit for fit in found[0] if contractions_cut(fit[1]) <= other]
        fitted, allowed = (kept, floored) if kept else (found[1], divisors)
        resident.update(max(fitted, key=lambda fit: fit[0])[1])
        indexed = {
            dim: sum(
                math.prod(t.shape) * t.operand_type.bitwidth
                for cn in cns
                for t in dict.fromkeys(cn.tensors)
                if dim in {lead(self.sub_workload.get_dims(cn)[p])[0] for p in map_dim_positions(cn.get_mapping(t))}
            )
            for dim in per_core
        }
        self._pipeline(resident, per_core, allowed, indexed, reductions, overflow, fill, aligned)
        if resident == seed:
            return []
        # A tiled reduction around tiled output dims keeps the outputs of every tile inside it resident while they
        # accumulate; where those alone do not fit, the reductions go innermost, finishing each output tile first.
        order = sorted((d for d in per_core if resident[d] < per_core[d]), key=lambda d: indexed.get(d, 0.0))
        outer = max((i for i, d in enumerate(order) if d in reductions), default=0)
        inside = frozenset(d for d in order[:outer] if d not in reductions)
        accumulators = tuple(cn for cn in cns if inside and order[outer] in reduced_by[cn])
        held = (
            self.memory_bits(accumulators, cores_per_node, unroll, resident, node_unroll, inside)
            if accumulators
            else {}
        )
        accumulating = any(bits > budgets[m] for m, bits in held.items() if budgets.get(m, 0) > 0)
        return self._emit(cns, resident, per_core, indexed, reductions if accumulating else set())

    def _fits(  # noqa: PLR0913
        self, seed, per_core, allowed, reductions, *, resident, overflow, fill, aligned, best_fill, seeded
    ) -> list[tuple[float, dict[LayerDim, int]]]:
        """The fitting tilings within ``allowed`` with their fill: the seed where it may and does fit, else shrunk."""
        resident.update(seed)
        if seeded and overflow() <= 0 and fill() >= best_fill:
            return [(fill(), dict(seed))]
        fitted = []
        for start in dict.fromkeys((tuple(seed.items()), tuple(per_core.items()))):
            for candidates in ([d for d in per_core if d not in reductions], list(per_core)):
                resident.update(start)
                while overflow() > 0 and (
                    step := self._shrink(candidates, resident, per_core, allowed, overflow, fill, aligned)
                ):
                    resident[step[0]] = step[1]
                if overflow() <= 0:
                    fitted.append((fill(), dict(resident)))
        return fitted

    def _pipeline(self, resident, per_core, divisors, indexed, reductions, overflow, fill, aligned) -> None:  # noqa: PLR0913
        """Cut ``resident`` into at least :attr:`PIPELINE_TILES` steady-state tiles, each cut the least that keeps the
        arrays as full, every tile whole or a multiple of the array's widest dim, and the memories within budget, along
        the dim indexing the most data (``indexed``, bits per dim), an output dim before a contraction."""
        full = fill()
        while math.prod(per_core[d] // resident[d] for d in per_core) < self.PIPELINE_TILES:
            steps = []
            for dim in per_core:
                keep = resident[dim]
                for smaller in (d for d in divisors[dim] if d < keep):
                    resident[dim] = smaller
                    if fill() >= full and aligned({dim}) and overflow() <= 0:
                        steps.append((indexed[dim], dim not in reductions, smaller / keep, dim, smaller))
                        break
                resident[dim] = keep
            if not steps:
                return
            *_, dim, smaller = max(steps, key=lambda step: step[:3])
            resident[dim] = smaller

    def _best_fill(self, cns, cores_per_node, unroll, per_core, node_unroll) -> float:
        """The array fill of whole per-core slices, which no tiling exceeds."""
        return self.array_fill(cns, cores_per_node, unroll, dict(per_core), node_unroll)

    def _shrink(  # noqa: PLR0913
        self, candidates, resident, per_core, divisors, overflow, fill, aligned
    ) -> tuple[LayerDim, int] | None:
        """The dim and smaller tile that keep the most of the arrays filled, a tile the array maps whole where one
        does, and, among those, save the most overflow per doubling of the steady-state tiles, within the tile budget;
        None when no step saves any. Looking past the next tile skips one that only starts double buffering what it
        shrinks."""
        base = overflow()
        best, best_key = None, (-math.inf, False, 0.0)
        for dim in candidates:
            keep = resident[dim]
            for smaller in (d for d in divisors[dim] if d < keep):
                resident[dim] = smaller
                if math.prod(per_core[d] / resident[d] for d in per_core) > self.MAX_STEADY_STATE_TILES:
                    break
                rate = (base - overflow()) / math.log2(keep / smaller)
                if rate > 0 and (key := (fill(), aligned({dim}), rate)) > best_key:
                    best, best_key = (dim, smaller), key
            resident[dim] = keep
        return best

    def _aligned(self, cns, cores_per_node, unroll, resident, node_unroll, per_core):  # noqa: PLR0913
        """A check that each node's extents in ``resident`` along the given dims are whole (``per_core``) or a multiple
        of its array's widest dim, which is all a tile can be cut to without the array mapping leaving part of that dim
        idle."""
        whole = {cn: self._node_sizes(cn, cores_per_node, unroll, dict(per_core), node_unroll) for cn in cns}

        def check(dims: set[LayerDim]) -> bool:
            for cn in cns:
                array = next(
                    (a for core in cores_per_node.get(cn, []) if (a := getattr(core, "operational_array", None))), None
                )
                if not array:
                    continue
                width = max(array.dimension_sizes.values())
                sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
                if any(sizes[d] % width and sizes[d] != whole[cn][d] for d in self._node_dims(cn) & dims):
                    return False
            return True

        return check

    def array_fill(
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None = None,
    ) -> float:
        """How much of their arrays the nodes can keep busy with these tiles, as the sum of each node's log fill: a
        node's tiles laid along its core's array dims, the largest along the largest, filling each up to its size."""
        total = 0.0
        for cn in cns:
            array = next(
                (a for core in cores_per_node.get(cn, []) if (a := getattr(core, "operational_array", None))), None
            )
            widths = sorted((w for w in array.dimension_sizes.values() if w > 1), reverse=True) if array else []
            if not widths:
                continue
            sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
            tiles = sorted((sizes[d] for d in self._node_dims(cn)), reverse=True) + [1] * len(widths)
            total += sum(math.log(min(tile, width) / width) for tile, width in zip(tiles, widths, strict=False))
        return total

    def _unrolled_extents(
        self, cns: tuple[ComputationNode, ...], cores_per_node: dict[ComputationNode, list[Core]]
    ) -> dict[LayerDim, int]:
        """Per unique dim, the least tile that keeps each node's array as full as its whole layer does."""
        from stream.stages.estimation.core_cost_backends import ZigZagBackend, select_backend  # noqa: PLC0415
        from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator  # noqa: PLC0415 -- import cycle

        extents: dict[LayerDim, int] = {}
        for cn in cns:
            core = next((c for c in cores_per_node.get(cn, []) if getattr(c, "operational_array", None)), None)
            if core is None or select_backend(core).name != ZigZagBackend.name:
                continue
            if (key := (cn.name, core.name)) not in self._unrolling:
                dims = self.sub_workload.get_dims(cn)
                try:
                    spread = ZigZagCostEstimator(self.sub_workload, self.accelerator, None).array_floors(cn, core)  # type: ignore[arg-type]
                except Exception as exc:  # noqa: BLE001 -- ZigZag rejects the layer, which its estimate then prices
                    logger.warning(
                        "No spatial mapping of %s on core %s (%s); its tiles keep no floor.", cn.name, core.id, exc
                    )
                    spread = {}
                self._unrolling[key] = {self.sub_workload.leading_dim(dims[p])[0]: f for p, f in spread.items()}
            for dim, factor in self._unrolling[key].items():
                extents[dim] = max(extents.get(dim, 1), factor)
        return extents

    def _node_sizes(
        self,
        cn: ComputationNode,
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
    ) -> dict[LayerDim, int]:
        """The extent of every unique dim on one core running ``cn`` when the group's cores hold ``resident``, and their
        whole slice along the dims it does not name: under its own split (``node_unroll``, else the group's ``unroll``
        on a multi-core node)."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        full = self._slices({})
        if node_unroll is not None:
            own = {lead(d)[0]: f for d, f in node_unroll.get(cn, {}).items()}
        else:
            own = (
                {d: f for d, f in unroll.items() if d in self._node_dims(cn)}
                if len(cores_per_node.get(cn, [])) > 1
                else {}
            )
        tiles = self._slices(unroll) | resident
        return {d: min(full[d], r * unroll.get(d, 1) // own.get(d, 1)) for d, r in tiles.items()}

    def _slices(self, unroll: dict[LayerDim, int]) -> dict[LayerDim, int]:
        """Each unique dim's extent on one core under the group's split ``unroll``; whole where it does not divide."""
        unique, _ = self.sub_workload.unique_dimensions()
        full = {z: self.sub_workload.get_dimension_size(z) for z in unique}
        return {z: n // u if (u := unroll.get(z, 1)) > 1 and n % u == 0 else n for z, n in full.items()}

    def memory_bits(
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None = None,
        accumulated_over: frozenset[LayerDim] = frozenset(),
    ) -> dict[str, float]:
        """Bits each memory holds when every unique dim of ``resident`` spans that tile on a core: per node a core
        runs, the tile of each operand the node touches under its own split (``node_unroll``, else the group's
        ``unroll`` on a multi-core node), windows included, a tensor the group produces once and one it reads from
        outside once per node reading it (the allocation gives each reader its own copy, but holds a copy in the memory
        of its source in place), twice where a temporal loop steps it (double buffered); cores sharing a memory fill it
        together. With ``accumulated_over``, only the outputs, held once, each spanning its whole per-core slice along
        those dims: what a reduction around them keeps resident while they accumulate."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        unique, _ = self.sub_workload.unique_dimensions()
        full = {z: self.sub_workload.get_dimension_size(z) for z in unique}
        held: dict[str, dict[tuple, float]] = {}
        produced = {t for cn in cns for t in cn.outputs}
        for cn in cns:
            cores = cores_per_node.get(cn, [])
            sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
            stepped = {d for d, r in resident.items() if r * unroll.get(d, 1) < full[d]}
            dims = self.sub_workload.get_dims(cn)
            if whole := {d: full[d] // unroll.get(d, 1) for d in accumulated_over}:
                sizes = self._node_sizes(cn, cores_per_node, unroll, resident | whole, node_unroll)
            for t in dict.fromkeys(cn.outputs if whole else cn.tensors):
                key = (t.name, cn.name, tuple(sizes[z] for z in unique))
                if (shape := self._shapes.get(key)) is None:
                    shape = self._shapes[key] = self.sub_workload.get_tensor_shape_with_dimension_sizes(t, sizes, cn)
                stepping = any(lead(dims[p])[0] in stepped for p in map_dim_positions(cn.get_mapping(t)))
                buffers = 2 if stepping and not whole else 1  # an accumulating output stays put
                tile = (t.name, shape) if t in produced else (t.name, cn.name, shape)
                for core in cores:
                    memory = self.accelerator.memory_of(core, role_of(cn, t)).name
                    held.setdefault(memory, {})[(core.id, tile)] = buffers * math.prod(shape) * t.operand_type.bitwidth
        return {memory: sum(tiles.values()) for memory, tiles in held.items()}

    def _capacities(
        self, cns: tuple[ComputationNode, ...], cores_per_node: dict[ComputationNode, list[Core]]
    ) -> dict[str, int]:
        """Each memory the group's cores use, with its capacity."""
        return {
            memory.name: memory.capacity_bits
            for cn in cns
            for core in cores_per_node.get(cn, [])
            for memory in self.accelerator.memories_of(core)
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
        indexed: dict[LayerDim, float],
        reductions: set[LayerDim],
    ) -> list[dict[str, Any]]:
        """One entry per tiled dim (resident tile < the per-core slice), naming a node that has the dim, else a node dim
        stepping along it, its tile scaled by the stride. The first entry is the innermost steady-state loop. The
        dims go from the one indexing the least data (``indexed``) to the most: what an inner loop does not index stays
        put while it runs, as a matmul's weights do over the rows of its activations, rather than being fetched again
        every iteration; ``reductions`` go innermost before them, so each output tile is complete before the next."""
        entries: list[dict[str, Any]] = []
        order = sorted(resident.items(), key=lambda item: (item[0] not in reductions, indexed.get(item[0], 0.0)))
        for dim, tile in order:
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
