"""Capacity-aware intra-core tiling for the generic auto-mapper."""

import logging
import math
from itertools import permutations
from typing import Any

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.mapping.mapping import placement_split_bits, role_of, whole_bits
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode, edge_tensor
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
    """Choose per-core tiles that fit each core's operand buffer within ``fill_fraction``, less what ``on_chip`` keeps
    there. Prefer tiles that walk the fewest placed bits and fill the arrays most, then cut them into at least
    ``PIPELINE_TILES`` without emptying the arrays."""

    MAX_STEADY_STATE_TILES = 1024
    """Steady-state tiles a group may be cut into, or, with ``scale_tiles``, per budget its whole per-core slices fill:
    what a group must stream through memories it overflows many times over."""
    MAX_ORDERED_LOOPS = 4
    """Tiled dims of a group up to which every order of its loops is compared for the data it moves; a group tiling
    more keeps the order of the data each dim indexes."""
    PIPELINE_TILES = 8
    """Steady-state tiles a group is cut into where its arrays stay as full, so that each tile's transfers overlap the
    others' compute rather than the whole group's loading, computing and storing following each other."""

    def __init__(
        self,
        sub_workload: Workload,
        accelerator: Accelerator,
        fill_fraction: float = 0.9,
        split_axes: dict[str, int] | None = None,
        nb_cols: int | None = None,
        on_chip: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.sub_workload = sub_workload
        self.nb_cols = nb_cols
        self.on_chip = on_chip or {}
        self.split_axes = split_axes or {}
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
        *,
        scale_tiles: bool = False,
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
        placed = self._placed(cns)

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
        max_tiles = (
            self._tile_limit(self.memory_bits(cns, cores_per_node, unroll, per_core, node_unroll), budgets)
            if scale_tiles
            else self.MAX_STEADY_STATE_TILES
        )
        fitted: list[tuple[float, dict[LayerDim, int]]] = []
        if (
            overflow() <= 0
            and fill() >= self._best_fill(cns, cores_per_node, unroll, per_core, node_unroll)
            and not any(placed.get(d) for d in per_core if seed[d] < per_core[d])
        ):
            fitted.append((fill(), dict(seed)))
        for start in dict.fromkeys((tuple(seed.items()), tuple(per_core.items()))) if not fitted else ():
            for candidates in ([d for d in per_core if d not in reductions], list(per_core)):
                resident.update(start)
                while overflow() > 0 and (
                    step := self._shrink(
                        candidates, resident, per_core, divisors, overflow, fill, aligned, placed, max_tiles
                    )
                ):
                    resident[step[0]] = step[1]
                if overflow() <= 0:
                    fitted.append((fill(), dict(resident)))
        if not fitted:
            return None

        def rank(fit: tuple[float, dict[LayerDim, int]]) -> tuple[int, float]:
            fill, tiles = fit
            return -sum(placed.get(d, 0) for d in per_core if tiles[d] < per_core[d]), fill

        resident.update(max(fitted, key=rank)[1])
        indexed = {
            dim: sum(
                math.prod(t.shape) * t.operand_type.bitwidth
                for cn in cns
                for t in dict.fromkeys(cn.tensors)
                if dim in self._indexing(cn, t)
            )
            for dim in per_core
        }
        self._pipeline(resident, per_core, divisors, indexed, reductions, overflow, fill, aligned, placed)
        if resident == seed:
            return []

        def accumulating(order: list[LayerDim]) -> bool:
            """Whether, looped in ``order``, a tiled reduction around tiled output dims keeps resident, while they
            accumulate, the outputs of every tile inside it, and those alone do not fit."""
            outer = max((i for i, d in enumerate(order) if d in reductions), default=0)
            inside = frozenset(d for d in order[:outer] if d not in reductions)
            accumulators = tuple(cn for cn in cns if inside and order[outer] in reduced_by[cn])
            held = (
                self.memory_bits(accumulators, cores_per_node, unroll, resident, node_unroll, inside)
                if accumulators
                else {}
            )
            return any(bits > budgets[m] for m, bits in held.items() if budgets.get(m, 0) > 0)

        order = sorted((d for d in per_core if resident[d] < per_core[d]), key=lambda d: indexed.get(d, 0.0))
        if accumulating(order):
            order.sort(key=lambda d: d not in reductions)
        feeding = [reduced_by[cn] for cn in cns if any(set(cn.outputs) & set(c.inputs) for c in cns)]
        orders = [order] + [
            list(o)
            for o in (permutations(order) if len(order) <= self.MAX_ORDERED_LOOPS else ())
            if list(o) != order
            and not accumulating(list(o))
            and all(o.index(d) < o.index(e) for r in feeding for d in r & set(o) for e in o if e not in r)
        ]
        return self._emit(
            cns, self._order(cns, cores_per_node, unroll, resident, node_unroll, budgets, orders), resident
        )

    def _order(  # noqa: PLR0913
        self, cns, cores_per_node, unroll, resident, node_unroll, budgets, orders
    ) -> list[LayerDim]:
        """The first of ``orders``, unless another moves fewer bits off chip in less time and no more bits into a memory
        with no link from off chip. Time is the largest bits per cycle of incoming links over all memories. The first
        is kept when the staging tiles alone overflow a budget."""
        if len(orders) == 1:
            return orders[0]
        held = self.memory_bits(cns, cores_per_node, unroll, resident, node_unroll)
        cores = {core for group in cores_per_node.values() for core in group}
        staging = [
            m
            for m in self.accelerator.staging_memories(self.nb_cols)
            if m.name not in budgets and self.accelerator.reaches(m, cores)
        ]
        staged = sum(
            2 * self._bits(cn, t, self._node_sizes(cn, cores_per_node, unroll, resident, {}))
            for cn, t in self._crossing(cns)
            if not self._is_on_chip(t)
        )
        capacities = self._capacities(cns, cores_per_node) | {m.name: self._room(m) for m in staging}
        tiles = {m: held.get(m, 0) for m in capacities} | {m.name: staged for m in staging}
        room = {m: budgets.get(m, int(c * self.fill_fraction)) - tiles[m] for m, c in capacities.items()}
        if any(free < 0 for free in room.values()):
            return orders[0]
        loose = {m: c - tiles[m] / 2 for m, c in capacities.items()}
        args = (cns, cores_per_node, unroll, resident, node_unroll)
        sources = {*self.accelerator.offchip_memories, *staging}
        rates: dict[str | None, float] = {
            m: sum(
                link.bandwidth
                for link in dict.fromkeys(
                    d["cl"] for u, _, d in self.accelerator.network.in_edges(memory, data=True) if u in sources
                )
            )
            for m in capacities
            for memory in [self.accelerator.memories[m]]
        }
        rates[None] = sum(self.accelerator.offchip_rate(m, "read") for m in self.accelerator.offchip_memories)

        def time(moved: dict[str | None, float]) -> float:
            return max((bits / rates[m] for m, bits in moved.items() if rates.get(m)), default=0)

        first = self._moved(*args, orders[0], loose, buffers=1, spills=True)
        better = [
            (time(moved), o)
            for o in orders[1:]
            if (moved := self._moved(*args, o, room)) is not None
            and moved[None] < first[None]
            and all(bits <= first.get(m, 0) for m, bits in moved.items() if not rates.get(m))
            and time(moved) < time(first)
        ]
        return min(better, key=lambda pair: pair[0])[1] if better else orders[0]

    def _crossing(self, cns: tuple[ComputationNode, ...]) -> list[tuple[ComputationNode, Tensor]]:
        """Each tensor the group reads from outside, per node reading it, and each it writes out, with its node."""
        produced = {t for cn in cns for t in cn.outputs}
        read = {t for cn in cns for t in cn.inputs}
        leaving = {edge_tensor(edge) for edge in self.sub_workload.get_out_edges()}
        return [
            (cn, t)
            for cn in cns
            for t in dict.fromkeys(cn.tensors)
            if (t in read and t not in produced) or (t in cn.outputs and (t not in read or t in leaving))
        ]

    def _moved(  # noqa: PLR0913
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
        order: list[LayerDim],
        room: dict[str, float],
        *,
        buffers: int = 2,
        spills: bool = False,
    ) -> dict[str | None, float] | None:
        """Bits moved into each memory (None is off chip) with the tiled dims looped in ``order``, innermost first. A
        tensor moves again on each iteration of an outer loop that does not index it, unless holding it fits ``room``.
        None if partial sums would leave the cores and ``spills`` is false."""
        operands = self._crossing(cns)
        read = {t for cn in cns for t in cn.inputs}
        made = [(cn, t) for cn in cns for t in cn.outputs if t in read]
        whole = self._slices(unroll)

        def bits(operand: tuple[ComputationNode, Tensor], sizes: dict[LayerDim, int], split: dict | None) -> int:
            cn, t = operand
            return self._bits(cn, t, self._node_sizes(cn, cores_per_node, unroll, sizes, split))

        staging = [m for m in room if self.accelerator.memories[m].staging]

        def places(readers: list[tuple[ComputationNode, Tensor]], spanned: dict[LayerDim, int]) -> list[dict]:
            """Where holding a tensor its ``readers`` read takes how much: on their cores, else in a staging memory."""
            copies: dict[str, float] = {}
            for operand in readers:
                for core in cores_per_node.get(operand[0], []):
                    memory = self.accelerator.memory_of(core, role_of(*operand)).name
                    if (share := self._placed_share(memory, operand[1])) == math.inf:
                        continue
                    extra = max(0, bits(operand, spanned, node_unroll) - share) - max(
                        0, buffers * bits(operand, resident, node_unroll) - share
                    )
                    copies[memory] = copies.get(memory, 0) + extra
            staged = sum(bits(o, spanned, {}) - buffers * bits(o, resident, {}) for o in readers)
            t = readers[0][1]
            staged_in = [{m: staged} for m in staging if t not in readers[0][0].outputs and not self._is_on_chip(t)]
            return [copies, *staged_in] if copies else staged_in

        left = dict(room)
        kept: dict[Tensor, dict[str, float]] = {}
        reads = {operand: [1, 1] for operand in dict.fromkeys([*operands, *made])}
        for i, dim in enumerate(order):
            inside = set(order[:i])
            spanned = resident | {d: whole[d] for d in inside}
            holding = [o for o in reads if dim not in self._indexing(*o) and inside & self._indexing(*o)]
            wanted = {
                t: places([o for o in holding if o[1] is t], spanned) for t in dict.fromkeys(t for _, t in holding)
            }
            least = {t: min((sum(o.values()) for o in options), default=0) for t, options in wanted.items()}
            for t, options in sorted(wanted.items(), key=lambda item: least[item[0]]):
                readers = [o for o in holding if o[1] is t]
                for memory, b in kept.get(t, {}).items():
                    left[memory] += b
                option = next((o for o in options if all(b <= left[m] for m, b in o.items())), None)
                if option is None and not spills and t in readers[0][0].outputs:
                    return None
                count = whole[dim] // resident[dim]
                for operand in readers:
                    reads[operand][0] *= count if option is None or any(m in staging for m in option) else 1
                    reads[operand][1] *= count if option is None else 1
                option = kept.get(t, {}) if option is None else option
                for memory, b in option.items():
                    left[memory] -= b
                kept[t] = option
        moved: dict[str | None, float] = {}
        offchip: dict[Tensor, float] = {}
        for operand in operands:
            cn, t = operand
            on_cores, off = reads[operand]
            output = t in cn.outputs
            if not self._is_on_chip(t):
                offchip[t] = max(offchip.get(t, 0), whole_bits(t) * (2 * off - 1 if output else off))
                for memory in staging:
                    moved[memory] = moved.get(memory, 0) + whole_bits(t) * (on_cores if output else off)
            for core in cores_per_node.get(cn, []):
                memory = self.accelerator.memory_of(core, role_of(cn, t)).name
                if (share := self._placed_share(memory, t)) == math.inf:
                    continue
                moved[memory] = moved.get(memory, 0) + max(0, bits(operand, whole, node_unroll) - share) * (
                    on_cores - 1 if output else on_cores
                )
        moved[None] = sum(offchip.values())
        return moved

    def _pipeline(self, resident, per_core, divisors, indexed, reductions, overflow, fill, aligned, placed) -> None:  # noqa: PLR0913
        """Cut ``resident`` into at least :attr:`PIPELINE_TILES` tiles, each cut the least that keeps the arrays as full
        and the memories within budget, every tile whole or a multiple of the array's widest dim, along the dim walking
        the fewest ``placed`` bits, then indexing the most data (``indexed``), an output dim before a contraction."""
        full = fill()
        while math.prod(per_core[d] // resident[d] for d in per_core) < self.PIPELINE_TILES:
            steps = []
            for dim in per_core:
                keep = resident[dim]
                for smaller in (d for d in divisors[dim] if d < keep):
                    resident[dim] = smaller
                    if fill() >= full and aligned({dim}) and overflow() <= 0:
                        steps.append(
                            (-placed.get(dim, 0), indexed[dim], dim not in reductions, smaller / keep, dim, smaller)
                        )
                        break
                resident[dim] = keep
            if not steps:
                return
            *_, dim, smaller = max(steps, key=lambda step: step[:4])
            resident[dim] = smaller

    def _tile_limit(self, whole: dict[str, float], budgets: dict[str, int]) -> int:
        """The most steady-state tiles a group may be cut into, its memories holding ``whole`` at whole slices."""
        filled = (math.ceil(bits / budgets[m]) for m, bits in whole.items() if budgets[m] > 0)
        return self.MAX_STEADY_STATE_TILES * max(1, *filled)

    def _placed(self, cns: tuple[ComputationNode, ...]) -> dict[LayerDim, int]:
        """Per dim, the bits of the group's operands a placement splits along an axis it indexes."""
        bits: dict[LayerDim, int] = {}
        for cn in cns:
            dims = self.sub_workload.get_dims(cn)
            for p, b in placement_split_bits(cn, self.split_axes).items():
                dim = self.sub_workload.leading_dim(dims[p])[0]
                bits[dim] = bits.get(dim, 0) + b
        return bits

    def _best_fill(self, cns, cores_per_node, unroll, per_core, node_unroll) -> float:
        """The array fill of whole per-core slices, which no tiling exceeds."""
        return self.array_fill(cns, cores_per_node, unroll, dict(per_core), node_unroll)

    def _shrink(  # noqa: PLR0913
        self, candidates, resident, per_core, divisors, overflow, fill, aligned, placed, max_tiles
    ) -> tuple[LayerDim, int] | None:
        """The dim and smaller tile that walk the fewest ``placed`` bits, fill the arrays most, map whole where one can,
        then save the most overflow per doubling of the tiles, within ``max_tiles``; None when none saves any. Walking a
        split axis makes each iteration wait on one memory, which costs more than a less filled array."""
        base = overflow()
        best, best_key = None, (-math.inf, -math.inf, False, 0.0)
        for dim in candidates:
            keep = resident[dim]
            for smaller in (d for d in divisors[dim] if d < keep):
                resident[dim] = smaller
                if math.prod(per_core[d] / resident[d] for d in per_core) > max_tiles:
                    break
                rate = (base - overflow()) / math.log2(keep / smaller)
                if rate > 0 and (key := (-placed.get(dim, 0), fill(), aligned({dim}), rate)) > best_key:
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
        """Bits each memory holds when every unique dim of ``resident`` spans that tile on a core: each node's operand
        tiles under its split, windows included, an outside input once per reader, twice where a loop steps it; cores
        sharing a memory fill it together. With ``accumulated_over``, only the outputs, whole along those dims."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        unique, _ = self.sub_workload.unique_dimensions()
        full = {z: self.sub_workload.get_dimension_size(z) for z in unique}
        held: dict[str, dict[tuple, float]] = {}
        tensors: dict[str, Tensor] = {}
        produced = {t for cn in cns for t in cn.outputs}
        for cn in cns:
            cores = cores_per_node.get(cn, [])
            sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
            stepped = {d for d, r in resident.items() if r * unroll.get(d, 1) < full[d]}
            if whole := {d: full[d] // unroll.get(d, 1) for d in accumulated_over}:
                sizes = self._node_sizes(cn, cores_per_node, unroll, resident | whole, node_unroll)
            for t in dict.fromkeys(cn.outputs if whole else cn.tensors):
                shape = self._shape(cn, t, sizes)
                stepping = bool(self._indexing(cn, t) & stepped)
                buffers = 2 if stepping and not whole else 1  # an accumulating output stays put
                tile = (t.name, shape) if t in produced else (t.name, cn.name, shape)
                tensors[t.name] = t
                for core in cores:
                    memory = self.accelerator.memory_of(core, role_of(cn, t)).name
                    held.setdefault(memory, {})[(core.id, tile)] = buffers * math.prod(shape) * t.operand_type.bitwidth
        return {memory: self._beyond_placed(memory, tiles, tensors) for memory, tiles in held.items()}

    def _beyond_placed(self, memory: str, tiles: dict[tuple, float], tensors: dict[str, Tensor]) -> float:
        """The bits ``tiles`` take in ``memory``, each tensor only beyond the block of it placed there."""
        per_tensor: dict[str, float] = {}
        for (_, (name, *_)), bits in tiles.items():
            per_tensor[name] = per_tensor.get(name, 0) + bits
        return sum(
            0 if (share := self._placed_share(memory, tensors[name])) == math.inf else max(0, bits - share)
            for name, bits in per_tensor.items()
        )

    def _placed_share(self, memory: str, t: Tensor) -> float:
        """The bits of ``t`` placed in ``memory`` for the whole run: 0 for none, ``inf`` where it is placed whole."""
        share = self.on_chip.get(memory, {}).get(t.name, 0.0)
        return math.inf if share >= whole_bits(t) else share

    def _indexing(self, cn: ComputationNode, t: Tensor) -> set[LayerDim]:
        """The unique dims ``cn`` indexes ``t`` along."""
        dims = self.sub_workload.get_dims(cn)
        return {self.sub_workload.leading_dim(dims[p])[0] for p in map_dim_positions(cn.get_mapping(t))}

    def _shape(self, cn: ComputationNode, t: Tensor, sizes: dict[LayerDim, int]) -> tuple[int, ...]:
        """The tile of ``t`` ``cn`` touches when its unique dims span ``sizes``."""
        unique, _ = self.sub_workload.unique_dimensions()
        key = (t.name, cn.name, tuple(sizes[z] for z in unique))
        if (shape := self._shapes.get(key)) is None:
            shape = self._shapes[key] = self.sub_workload.get_tensor_shape_with_dimension_sizes(t, sizes, cn)
        return shape

    def _bits(self, cn: ComputationNode, t: Tensor, sizes: dict[LayerDim, int]) -> int:
        """The bits of :meth:`_shape`."""
        return math.prod(self._shape(cn, t, sizes)) * t.operand_type.bitwidth

    def _capacities(
        self, cns: tuple[ComputationNode, ...], cores_per_node: dict[ComputationNode, list[Core]]
    ) -> dict[str, int]:
        """Each memory the group's cores use, with the room it has (:meth:`_room`)."""
        return {
            memory.name: self._room(memory)
            for cn in cns
            for core in cores_per_node.get(cn, [])
            for memory in self.accelerator.memories_of(core)
        }

    def _room(self, memory: Memory) -> float:
        """The capacity of ``memory`` less the workload inputs and outputs placed in it for the whole run."""
        return memory.capacity_bits - sum(self.on_chip.get(memory.name, {}).values())

    def _is_on_chip(self, t: Tensor) -> bool:
        """Whether ``t`` is placed in an on-chip memory for the whole run, so it moves no data off chip."""
        return any(t.name in tensors for tensors in self.on_chip.values())

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
        self, cns: tuple[ComputationNode, ...], order: list[LayerDim], resident: dict[LayerDim, int]
    ) -> list[dict[str, Any]]:
        """One entry per tiled dim of ``order``, innermost steady-state loop first, naming a node that has the dim, else
        a node dim stepping along it, its tile scaled by the stride."""
        entries: list[dict[str, Any]] = []
        for dim in order:
            tile = resident[dim]
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
