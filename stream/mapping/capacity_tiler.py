"""Capacity-aware intra-core tiling for the generic auto-mapper."""

import logging
import math
from collections.abc import Sequence
from itertools import permutations
from typing import Any

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.cost import running_sum_bits
from stream.hardware.ports import READ
from stream.mapping.mapping import placement_split_bits, role_of, whole_bits
from stream.opt.allocation.constraint_optimization.hardware import build_hardware_facts
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode, edge_tensor
from stream.workload.steady_state.iteration_space import IterationVariable, IterationVariableType, LoopEffect
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
    """Choose per-core tiles that fit ``fill_fraction`` of each core's operand buffer, less what the toolchain reserves
    and ``on_chip`` keeps there. Prefer tiles that walk the fewest placed bits and fill the arrays most, then cut them
    into at least ``PIPELINE_TILES`` without emptying the arrays."""

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
        fill_fraction: float = 1.0,
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
        self.hardware = build_hardware_facts(accelerator, self.nb_cols)
        self._shapes: dict[tuple, tuple[int, ...]] = {}

    def plan(  # noqa: PLR0913, PLR0915
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
        budgets = self._capacities(cns, cores_per_node)

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
        args = (cns, cores_per_node, unroll, resident, node_unroll)

        def overflow() -> float:
            """Largest amount by which any memory exceeds its budget (0 means the whole group fits). Never negative, so
            a shrink gets no credit for going past fitting."""
            held = self.memory_bits(cns, cores_per_node, unroll, resident, node_unroll)
            return max([0.0, *(bits - budgets[m] for m, bits in held.items() if budgets[m] > 0)])

        def fill() -> float:
            return self.array_fill(cns, cores_per_node, unroll, resident, node_unroll)

        def folded() -> float:
            return self.array_fill(*args, folded=True)

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
        indexed = self._indexed(cns, per_core)
        feeding = [reduced_by[cn] for cn in cns if any(set(cn.outputs) & set(c.inputs) for c in cns)]

        def accumulating(order: list[LayerDim]) -> bool:
            """Whether, looped in ``order``, a tiled reduction around tiled output dims keeps resident, while they
            accumulate, the outputs of every tile inside it, and the group then does not fit."""
            outer = max((i for i, d in enumerate(order) if d in reductions), default=0)
            inside = frozenset(d for d in order[:outer] if d not in reductions)
            accumulators = tuple(cn for cn in cns if inside and order[outer] in reduced_by[cn])
            held = self.memory_bits(*args, accumulators, inside) if accumulators else {}
            return any(bits > budgets[m] for m, bits in held.items() if budgets.get(m, 0) > 0)

        def default_order() -> list[LayerDim]:
            return sorted((d for d in per_core if resident[d] < per_core[d]), key=lambda d: indexed.get(d, 0.0))

        def loop_orders() -> list[list[LayerDim]]:
            order = default_order()
            if accumulating(order):
                order.sort(key=lambda d: d not in reductions)
            return [order] + [
                list(o)
                for o in (permutations(order) if len(order) <= self.MAX_ORDERED_LOOPS else ())
                if list(o) != order
                and not accumulating(list(o))
                and all(o.index(d) < o.index(e) for r in feeding for d in r & set(o) for e in o if e not in r)
            ]

        def offchip() -> float:
            """Bits the group moves off chip, looped in the order moving the fewest."""
            held = self.memory_bits(*args)
            room = {m: budget - held.get(m, 0) for m, budget in budgets.items()}
            return min(self._moved(*args, order, room, spills=True)[None] for order in loop_orders())

        def pipelined() -> tuple[float, float, float]:
            return self._pipelined_estimate(*args, budgets, per_core, self._order(*args, budgets, loop_orders()))

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
        fits_whole = bool(fitted)

        search = {
            "resident": resident,
            "per_core": per_core,
            "divisors": divisors,
            "overflow": overflow,
            "fill": fill,
            "aligned": aligned,
            "placed": placed,
        }
        fitted = fitted or self._fit(
            (seed, per_core), [d for d in per_core if d not in reductions], max_tiles, offchip, **search
        )
        if not fitted:
            return None

        def rank(fit: tuple[float, dict[LayerDim, int]]) -> tuple[int, float]:
            fill, tiles = fit
            return -sum(placed.get(d, 0) for d in per_core if tiles[d] < per_core[d]), fill

        self._settle(fitted, rank, indexed, reductions, **search)
        if fits_whole and (streams := self._overlap_dims(cns, unroll, node_unroll) - reductions):
            # A group that fits whole and computes longer than it transfers is cut if that takes less time.
            before, compute, transfer = pipelined()
            if compute > transfer and math.prod(per_core[d] // resident[d] for d in per_core) < self.PIPELINE_TILES:
                uncut = dict(resident)
                cuts = {d: sizes if d in streams else [] for d, sizes in divisors.items()}
                self._pipeline(resident, per_core, cuts, indexed, reductions, overflow, folded, lambda _: True, placed)
                if pipelined()[0] >= before:
                    resident.update(uncut)
        if resident == seed:
            return []
        if accumulating(default_order()) and (
            refitted := self._fit((per_core,), [d for d in reductions if d in per_core], max_tiles, offchip, **search)
        ):
            self._settle(refitted, rank, indexed, reductions, **search)
        return self._emit(cns, self._order(*args, budgets, loop_orders()), resident)

    def _overlap_dims(
        self,
        cns: tuple[ComputationNode, ...],
        unroll: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
    ) -> set[LayerDim]:
        """The dims every node of the group has and no node splits over cores."""
        lead = self.sub_workload.leading_dim
        split = {lead(d)[0] for f in [unroll, *(node_unroll or {}).values()] for d, n in f.items() if n > 1}
        return set.intersection(*(self._node_dims(cn) for cn in cns)) - split

    def _indexed(self, cns: tuple[ComputationNode, ...], per_core: dict[LayerDim, int]) -> dict[LayerDim, float]:
        """Per dim, the bits of the group's tensors it indexes."""
        return {
            dim: sum(
                math.prod(t.shape) * t.operand_type.bitwidth
                for cn in cns
                for t in dict.fromkeys(cn.tensors)
                if dim in self._indexing(cn, t)
            )
            for dim in per_core
        }

    def _fit(  # noqa: PLR0913
        self,
        starts: tuple[dict[LayerDim, int], ...],
        first: list[LayerDim],
        max_tiles: int,
        offchip: Any,
        *,
        resident: dict[LayerDim, int],
        per_core: dict[LayerDim, int],
        divisors: dict[LayerDim, list[int]],
        overflow: Any,
        fill: Any,
        aligned: Any,
        placed: dict[LayerDim, int],
    ) -> list[tuple[float, dict[LayerDim, int]]]:
        """The tilings that fit, with their array fill, that shrinking ``resident`` from each of ``starts`` reaches,
        cutting the dims ``first`` alone before all of them."""
        before, found = dict(resident), []
        for start in dict.fromkeys(tuple(start.items()) for start in starts):
            for candidates in (first, list(per_core)):
                resident.update(start)
                while overflow() > 0 and (
                    step := self._shrink(
                        candidates, resident, per_core, divisors, overflow, fill, aligned, placed, offchip, max_tiles
                    )
                ):
                    resident[step[0]] = step[1]
                if overflow() <= 0:
                    found.append((fill(), dict(resident)))
        if not found:
            resident.update(before)
        return found

    def _settle(  # noqa: PLR0913
        self,
        fits: list[tuple[float, dict[LayerDim, int]]],
        rank: Any,
        indexed: dict[LayerDim, float],
        reductions: set[LayerDim],
        *,
        resident: dict[LayerDim, int],
        per_core: dict[LayerDim, int],
        divisors: dict[LayerDim, list[int]],
        overflow: Any,
        fill: Any,
        aligned: Any,
        placed: dict[LayerDim, int],
    ) -> None:
        """Set ``resident`` to the best of ``fits`` by ``rank``, cut into tiles enough to pipeline."""
        resident.update(max(fits, key=rank)[1])
        self._pipeline(resident, per_core, divisors, indexed, reductions, overflow, fill, aligned, placed)

    def _pipelined_estimate(  # noqa: PLR0913
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
        budgets: dict[str, float],
        per_core: dict[LayerDim, int],
        order: list[LayerDim],
    ) -> tuple[float, float, float]:
        """Estimated cycles of the group in the n tiles of ``resident``, with its compute cycles C and its transfer
        cycles: X0 for what no tile loop indexes, moved once, then C and the rest X overlapping but for one tile,
        X0 + (C + X) / n + (n - 1) / n * max(C, X)."""
        n = math.prod(per_core[d] // resident[d] for d in per_core)
        args = (cns, cores_per_node, unroll, resident, node_unroll)
        capacities, tiles, rates = self._transfers(*args, budgets)
        room = {m: c - tiles[m] for m, c in capacities.items()}
        moved = self._moved(*args, order, room, spills=True)
        fixed = {t for cn, t in self._crossing(cns) if not self._indexing(cn, t) & set(order)}
        once = self._moved(*args, [], room, spills=True, tensors=fixed)
        alone = self._link_cycles(once, rates)
        transfer = self._link_cycles({m: bits - once.get(m, 0) for m, bits in moved.items()}, rates)
        compute = max(self.compute_cycles(*args), self._input_port_cycles(*args, n))
        return alone + (compute + transfer) / n + (n - 1) / n * max(compute, transfer), compute, alone + transfer

    def _input_port_cycles(  # noqa: PLR0913
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
        tiles: int,
    ) -> float:
        """Cycles the busiest port a core reads its inputs from takes to read each input tile once per
        each of ``tiles`` tiles."""
        reads: dict[Any, float] = {}
        for cn in cns:
            sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
            for t in dict.fromkeys(cn.inputs):
                for core in cores_per_node.get(cn, []):
                    role = role_of(cn, t)
                    port = self.accelerator.ports.port_for(core, READ, role, self.accelerator.memory_of(core, role))
                    if port is not None:
                        reads[port] = reads.get(port, 0.0) + tiles * self._bits(cn, t, sizes) / port.bits_per_cycle
        return max(reads.values(), default=0.0)

    def _transfers(  # noqa: PLR0913
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
        budgets: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str | None, float]]:
        """The budget of each memory the tiles pass through, the bits of tiles it holds, and the bits per cycle of its
        links from off chip or a staging memory (None: off chip's own read rate)."""
        held = self.memory_bits(cns, cores_per_node, unroll, resident, node_unroll)
        cores = {core for group in cores_per_node.values() for core in group}
        staging = [
            m
            for m in self.accelerator.staging_memories(self.nb_cols)
            if m.name not in budgets and self.accelerator.reaches(m, cores)
        ]
        staged = sum(
            buffer_count(1, True) * self._bits(cn, t, self._node_sizes(cn, cores_per_node, unroll, resident, {}))
            for cn, t in self._crossing(cns)
            if not self._is_on_chip(t)
        )
        capacities = budgets | {m.name: self._budget(m) for m in staging}
        tiles = {m: held.get(m, 0) for m in capacities} | {m.name: staged for m in staging}
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
        return capacities, tiles, rates

    @staticmethod
    def _link_cycles(moved: dict[str | None, float], rates: dict[str | None, float]) -> float:
        """Cycles the busiest incoming links take to carry ``moved``, at ``rates``."""
        return max((bits / rates[m] for m, bits in moved.items() if rates.get(m)), default=0)

    def _order(  # noqa: PLR0913
        self, cns, cores_per_node, unroll, resident, node_unroll, budgets, orders
    ) -> list[LayerDim]:
        """The first of ``orders``, unless another that fits moves fewer bits off chip in less time and no more bits
        into a memory with no link from off chip. Time is the largest bits per cycle of incoming links over all
        memories. The first is kept when the staging tiles alone overflow a budget."""
        if len(orders) == 1:
            return orders[0]
        args = (cns, cores_per_node, unroll, resident, node_unroll)

        def fits(order: list[LayerDim]) -> bool:
            held = self.memory_bits(*args, order=order)
            return all(bits <= budgets[m] for m, bits in held.items() if budgets.get(m, 0) > 0)

        capacities, tiles, rates = self._transfers(*args, budgets)
        room = {m: c - tiles[m] for m, c in capacities.items()}
        if any(free < 0 for free in room.values()):
            return orders[0]
        loose = {m: c - tiles[m] / 2 for m, c in capacities.items()}
        first = self._moved(*args, orders[0], loose, buffers=1, spills=True)
        better = [
            (self._link_cycles(moved, rates), o)
            for o in orders[1:]
            if fits(o)
            and (moved := self._moved(*args, o, room)) is not None
            and moved[None] < first[None]
            and all(bits <= first.get(m, 0) for m, bits in moved.items() if not rates.get(m))
            and self._link_cycles(moved, rates) < self._link_cycles(first, rates)
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
        tensors: set[Tensor] | None = None,
    ) -> dict[str | None, float] | None:
        """Bits moved into each memory (None is off chip) with the tiled dims looped in ``order``, innermost first, of
        ``tensors`` (all where None). A tensor moves again on each iteration of an outer loop that does not index it,
        unless holding it fits ``room``. None if partial sums would leave the cores and ``spills`` is false."""
        operands = [o for o in self._crossing(cns) if tensors is None or o[1] in tensors]
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

    def _tile_limit(self, whole: dict[str, float], budgets: dict[str, float]) -> int:
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
        self, candidates, resident, per_core, divisors, overflow, fill, aligned, placed, offchip, max_tiles
    ) -> tuple[LayerDim, int] | None:
        """The dim and smaller tile that walk the fewest ``placed`` bits, map whole where one can, then save the most
        overflow per doubling of the tiles and of the compute time (through the ``fill`` lost), within ``max_tiles``;
        of ties that each make the group fit, the one moving the fewest bits off chip. None when none saves any."""
        base, full = overflow(), fill()
        best: list[tuple[LayerDim, int]] = []
        best_key, fitting = (-math.inf, False, 0.0), False
        for dim in candidates:
            keep = resident[dim]
            for smaller in (d for d in divisors[dim] if d < keep):
                resident[dim] = smaller
                if math.prod(per_core[d] / resident[d] for d in per_core) > max_tiles:
                    break
                left = overflow()
                rate = (base - left) / (math.log2(keep / smaller) + (full - fill()) / math.log(2))
                key = (-placed.get(dim, 0), aligned({dim}), rate)
                if rate > 0 and key > best_key:
                    best, best_key, fitting = [(dim, smaller)], key, left <= 0
                elif rate > 0 and key == best_key and fitting and left <= 0:
                    best.append((dim, smaller))
            resident[dim] = keep
        if not best:
            return None
        if len(best) == 1:
            return best[0]

        def moved(step: tuple[LayerDim, int]) -> float:
            keep = resident[step[0]]
            resident[step[0]] = step[1]
            bits = offchip()
            resident[step[0]] = keep
            return bits

        return min(best, key=moved)

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
        *,
        folded: bool = False,
    ) -> float:
        """How much of their arrays the nodes can keep busy with these tiles, as the sum of each node's log fill."""
        fills = (self._node_fill(cn, cores_per_node, unroll, resident, node_unroll, folded=folded) for cn in cns)
        return sum(found[0] for found in fills if found)

    def compute_cycles(
        self,
        cns: tuple[ComputationNode, ...],
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None = None,
    ) -> float:
        """Cycles the nodes' arrays take for the whole group with these tiles: each node's MACs over its cores, its
        array's units and its fill."""
        total = 0.0
        for cn in cns:
            if found := self._node_fill(cn, cores_per_node, unroll, resident, node_unroll):
                log_fill, units = found
                macs = math.prod(self.sub_workload.get_dimension_size(d) for d in self.sub_workload.get_dims(cn))
                total += macs / (len(cores_per_node[cn]) * units * math.exp(log_fill))
        return total

    def _node_fill(
        self,
        cn: ComputationNode,
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
        *,
        folded: bool = False,
    ) -> tuple[float, int] | None:
        """The log fill of ``cn``'s array and the array's units (None: no array): a node's tiles laid along its core's
        array dims, the largest along the largest, filling each up to its size or, ``folded``, in passes of its size."""
        array = next(
            (a for core in cores_per_node.get(cn, []) if (a := getattr(core, "operational_array", None))), None
        )
        widths = sorted((w for w in array.dimension_sizes.values() if w > 1), reverse=True) if array else []
        if not widths:
            return None
        sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
        tiles = sorted((sizes[d] for d in self._node_dims(cn)), reverse=True) + [1] * len(widths)
        fill = sum(
            math.log(tile / (math.ceil(tile / width) * width) if folded else min(tile, width) / width)
            for tile, width in zip(tiles, widths, strict=False)
        )
        return fill, math.prod(widths)

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
        own = self._own_split(cn, cores_per_node, unroll, node_unroll)
        tiles = self._slices(unroll) | resident
        return {d: min(full[d], r * unroll.get(d, 1) // own.get(d, 1)) for d, r in tiles.items()}

    def _own_split(
        self,
        cn: ComputationNode,
        cores_per_node: dict[ComputationNode, list[Core]],
        unroll: dict[LayerDim, int],
        node_unroll: dict[ComputationNode, dict[LayerDim, int]] | None,
    ) -> dict[LayerDim, int]:
        """The split of each unique dim over the cores running ``cn``: its own (``node_unroll``), else the group's
        ``unroll`` on a multi-core node."""
        lead = self.sub_workload.leading_dim
        if node_unroll is not None:
            return {lead(d)[0]: f for d, f in node_unroll.get(cn, {}).items()}
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        multi = len(cores_per_node.get(cn, [])) > 1
        return {d: f for d, f in unroll.items() if d in self._node_dims(cn)} if multi else {}

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
        accumulating: tuple[ComputationNode, ...] = (),
        accumulated_over: frozenset[LayerDim] = frozenset(),
        order: Sequence[LayerDim] | None = None,
    ) -> dict[str, float]:
        """Bits each memory holds when every unique dim of ``resident`` spans that tile on a core: an outside input once
        per reader, a produced tensor at its producer's tiles in the memory and each reader's windows there beyond them,
        twice where a loop steps it, plus the halo the tile loops keep (looped in ``order``); cores sharing a memory
        fill it together. Outputs of ``accumulating`` are whole along ``accumulated_over``."""
        lead = self.sub_workload.leading_dim
        unroll = {lead(d)[0]: f for d, f in unroll.items()}
        unique, _ = self.sub_workload.unique_dimensions()
        full = {z: self.sub_workload.get_dimension_size(z) for z in unique}
        loops = self._loops(cns, unroll, resident, order)
        held: dict[str, dict[tuple, float]] = {}
        tensors: dict[str, Tensor] = {}
        producers = {t: cn for cn in cns for t in cn.outputs}
        for cn in cns:
            cores = cores_per_node.get(cn, [])
            sizes = self._node_sizes(cn, cores_per_node, unroll, resident, node_unroll)
            stepped = {d for d, r in resident.items() if r * unroll.get(d, 1) < full[d]}
            split = {d for d, f in self._own_split(cn, cores_per_node, unroll, node_unroll).items() if f > 1}
            running = running_sum_bits(self.sub_workload, cn, cores, stepped, split)
            whole = {d: full[d] // unroll.get(d, 1) for d in accumulated_over} if cn in accumulating else {}
            spanned = self._node_sizes(cn, cores_per_node, unroll, resident | whole, node_unroll) if whole else sizes
            for t in dict.fromkeys(cn.tensors):
                kept = bool(whole) and t in cn.outputs
                shape = self._shape(cn, t, spanned if kept else sizes)
                stepping = bool(self._indexing(cn, t) & stepped)
                from_producer = t in cn.inputs and producers.get(t, cn) is not cn
                stop = self._held_stop(cn, t, producers[t], loops) if from_producer else -1
                buffers = (
                    self._held_windows(cn, t, loops, stop) if stop >= 0 else buffer_count(1, stepping and not kept)
                )
                tensors[t.name] = t
                is_running_sum = bool(running) and t in cn.outputs
                width = running if is_running_sum else t.operand_type.bitwidth
                bits = buffers * math.prod(shape) * width
                if t in cn.inputs and not kept:
                    split = self._own_split(cn, cores_per_node, unroll, node_unroll)
                    bits += self._kept_halo(cn, t, sizes, split, loops, stop) * width
                for core in cores:
                    tiles = held.setdefault(self.accelerator.memory_of(core, role_of(cn, t)).name, {})
                    tiles[(t.name, cn.name)] = tiles.get((t.name, cn.name), 0.0) + bits
        made = {t.name: cn.name for t, cn in producers.items()}
        return {memory: self._beyond_placed(memory, tiles, tensors, made) for memory, tiles in held.items()}

    def _loops(
        self,
        cns: tuple[ComputationNode, ...],
        unroll: dict[LayerDim, int],
        resident: dict[LayerDim, int],
        order: Sequence[LayerDim] | None,
    ) -> list[tuple[LayerDim, int]]:
        """The group's tile loops, innermost first, each with its trips: ``order``, else the dims indexing the fewest
        bits innermost, as :meth:`plan` loops them by default."""
        slices = self._slices(unroll)
        cut = {d: slices[d] // r for d, r in resident.items() if r < slices[d]}
        if order is None:
            indexed = self._indexed(cns, cut)
            order = sorted(cut, key=lambda d: indexed.get(d, 0.0))
        return [(d, cut[d]) for d in order if d in cut]

    def _held_stop(
        self, cn: ComputationNode, t: Tensor, producer: ComputationNode, loops: list[tuple[LayerDim, int]]
    ) -> int:
        """The outermost of ``loops`` (innermost first) across which ``cn`` keeps its copy of ``t``, as ``producer``
        does not make a new tile there; -1 for none."""
        moves = self._indexing(producer, t)
        stepped = self._node_dims(cn) | self._node_dims(producer)
        return max((i for i, (d, _) in enumerate(loops) if d in stepped and d not in moves), default=-1)

    def _held_windows(self, cn: ComputationNode, t: Tensor, loops: list[tuple[LayerDim, int]], stop: int) -> int:
        """Windows of ``t`` ``cn`` keeps across ``loops`` up to ``stop``, doubled where an outer loop moves them."""
        moving = self._indexing(cn, t)
        windows = math.prod(trips for d, trips in loops[: stop + 1] if d in moving)
        return buffer_count(windows, any(d in moving for d, _ in loops[stop + 1 :]))

    def _kept_halo(  # noqa: PLR0913
        self,
        cn: ComputationNode,
        t: Tensor,
        sizes: dict[LayerDim, int],
        split: dict[LayerDim, int],
        loops: list[tuple[LayerDim, int]],
        stop: int = -1,
    ) -> int:
        """Elements of halo ``cn`` keeps of ``t`` in addition to its windows at reuse stop ``stop`` while the tile
        ``loops`` slide it, as the allocation counts them."""
        dims = set(self.sub_workload.get_tensor_dimensions(t))
        windows = self.sub_workload.windows_for_sizes(t, cn, sizes, [d for d, _ in loops if d in dims], split)
        if not windows:
            return 0
        variables = [
            IterationVariable(
                dimension=d,
                size=trips,
                effect=LoopEffect.VARYING if d in dims else LoopEffect.INVARIANT,
                type=IterationVariableType.TEMPORAL,
                halo=windows[d][1] if d in windows else 0,
            )
            for d, trips in loops
        ]
        return self.sub_workload.kept_halos_for_sizes(t, cn, windows, sizes, variables).get(stop, 0)

    def _beyond_placed(
        self, memory: str, tiles: dict[tuple[str, str], float], tensors: dict[str, Tensor], made: dict[str, str]
    ) -> float:
        """The bits ``tiles`` take in ``memory``: a tensor's tile at the node that ``made`` it, each reader's copy only
        beyond that, and each tensor only beyond the block of it placed there."""
        per_tensor: dict[str, float] = {}
        for (name, node), bits in tiles.items():
            source = tiles.get((name, made[name]), 0.0) if name in made else 0.0
            per_tensor[name] = per_tensor.get(name, 0.0) + (bits if node == made.get(name) else max(0.0, bits - source))
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
    ) -> dict[str, float]:
        """Each memory the group's cores use, with the budget its tiles may fill (:meth:`_budget`)."""
        return {
            memory.name: self._budget(memory)
            for cn in cns
            for core in cores_per_node.get(cn, [])
            for memory in self.accelerator.memories_of(core)
        }

    def _budget(self, memory: Memory) -> float:
        """The ``fill_fraction`` of :meth:`_room` the tiles may fill."""
        return self.fill_fraction * self._room(memory)

    def _room(self, memory: Memory) -> float:
        """The capacity of ``memory`` less what the toolchain reserves and the workload inputs and outputs placed in it
        for the whole run."""
        return self.hardware.usable_bits(memory) - sum(self.on_chip.get(memory.name, {}).values())

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


def buffer_count(tiles: int, double_buffered: bool) -> int:
    """Buffers a memory keeps for ``tiles`` tiles of a tensor: at least two where it is double-buffered, so the next
    tile lands while the last is read."""
    return max(tiles, 2) if double_buffered else tiles
