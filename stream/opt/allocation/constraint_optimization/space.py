"""The choices an allocation model decides between, read off the steady-state problem before any variable exists."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from functools import cached_property, reduce
from math import ceil, prod
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.communication_manager import Hop, MulticastPathPlan
from stream.datatypes import ELEMENT_BITS
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.hardware.cost import (
    RunningSumCall,
    kept_in_output_buffer,
    partial_sum_format,
    reduced_dims,
    running_sum_bits,
    running_sum_format,
)
from stream.hardware.ports import OUTPUT, READ
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.capacity_tiler import buffer_count
from stream.mapping.mapping import placements_of, role_of, the_one
from stream.opt.allocation.constraint_optimization.diagnosis import structural_infeasibility
from stream.opt.allocation.constraint_optimization.layouts import CopyLayouts
from stream.opt.allocation.constraint_optimization.utils import (
    absent_trips,
    get_active_latency,
    get_transfer_latency_for_path,
)
from stream.workload.demand import (
    Box,
    DemandItem,
    box_bits,
    copy_sets,
    covered,
    firing_count,
    item_bits,
    moved_elements,
)
from stream.workload.iterator_type import is_state_operand
from stream.workload.node import HasOutputs, TransferType
from stream.workload.steady_state.iteration_space import IterationVariableType, LoopEffect, Reuse
from stream.workload.workload import ComputationNode, HasIterationSpace, InEdge, OutEdge, Tensor, TransferNode

if TYPE_CHECKING:
    from stream.allocation.problem import AllocationProblem
    from stream.cost_model.layout import Rate
    from stream.datatypes import LayerDim
    from stream.hardware.architecture.backends.zigzag import OutputBufferUse
    from stream.hardware.ports import Port
    from stream.workload.steady_state.node import Node
    from stream.workload.workload import Workload

CopySet: TypeAlias = tuple[int, ...]
"""The copies of a transfer that read a piece of its demand, as indices into its outputs."""
Placement: TypeAlias = tuple[Memory, ...]
"""The memory a tensor sits in at each position of its node; cores sharing a memory repeat it."""
Choice: TypeAlias = tuple[TransferNode, MulticastPathPlan]


class DecisionSpace:
    """What the allocation decides between: the placements each tensor may take, the routes each transfer may
    take and the reuse stops of each tensor, with the facts every family reads off them. Read-only once built."""

    def __init__(self, problem: AllocationProblem) -> None:
        self.problem = problem
        self.workload = problem.workload
        self.slot_of = problem.timeslots
        self.accelerator = problem.accelerator
        self.hardware = problem.hardware
        self.shared_bandwidth: dict[Memory, BandwidthModel] = dict(self.accelerator.bandwidth)
        self.iterations = problem.iterations
        self.ssis = problem.ssis
        self.mapping = problem.mapping
        self.cost_lut = problem.cost_lut
        self.max_slot = max(self.slot_of.values()) if self.slot_of else 0
        self.big_m = len(self.workload.nodes()) + 5
        self.mem_memories = list(self.hardware.mem_memories)
        self.ssc_nodes: tuple[ComputationNode, ...] = tuple(self.workload.get_computation_nodes())
        self.transfer_nodes: tuple[TransferNode, ...] = tuple(self.workload.get_transfer_nodes())

        self.tensor_fixed: list[Tensor] = []
        self.tensor_var: list[Tensor] = []
        self.tensor_choices: dict[Tensor, tuple[Placement, ...]] = {}
        self.path_choices: dict[TransferNode, tuple[MulticastPathPlan, ...]] = {}
        self.partial_sum_ratio: dict[TransferNode, float] = {}
        self._init_option_sets()
        self._fixed = frozenset(self.tensor_fixed)
        self._candidates: dict[Tensor, set[Memory]] = {}
        self._firing_cycles: dict[tuple[TransferNode, MulticastPathPlan, CopySet | None], int] = {}
        self._side_ports: dict[tuple[object, ...], dict[Port, float]] = {}
        self._broadcast: dict[TransferNode, bool] = {}
        self._one_memory: dict[TransferNode, bool] = {}
        self._overlaps: dict[TransferNode, dict[tuple[int, int], float]] = {}
        self._compute_cores: dict[TransferNode, frozenset[Core]] = {}

        self.reuse_levels: dict[tuple[Tensor, int], int] = {}
        self.tiles_needed_levels: dict[tuple[Tensor, int], int] = {}
        self.rotation_levels: dict[tuple[Tensor, int], bool] = {}
        self.bds_needed_levels: dict[tuple[Tensor, int], int] = {}
        self.tensors_to_optimize_reuse_for: list[Tensor] = []
        self._ensure_same_ssis_for_all_transfers()
        self._init_transfer_fire_helpers()

        self.link_set: set[CommunicationLink] = set()
        self.links_in_choice: dict[Choice, set[CommunicationLink]] = {}
        self.choice_src_memories: dict[Choice, set[Memory]] = {}
        self.choice_dst_memories: dict[Choice, set[Memory]] = {}
        self.choice_has_empty_path: dict[Choice, bool] = {}
        self._index_choice_metadata()

    def _init_option_sets(self) -> None:
        for node in self.workload.topological_sort():
            if not isinstance(node, HasOutputs):
                continue
            carried = (
                [x for x in node.inputs if is_state_operand(node, x)] if isinstance(node, HasIterationSpace) else []
            )
            for tensor in (*node.outputs, *carried):
                if tensor in self.tensor_choices:
                    continue
                try:
                    normalized = _normalize_tensor_choices(self.placements(node, tensor))
                    if isinstance(node, TransferNode):
                        normalized = self._reader_placements(node, tensor, normalized)
                except ValueError as exc:
                    raise InfeasibleAllocationError(
                        structural_infeasibility(
                            f"node '{getattr(node, 'name', node)}' has no memory it can hold {tensor.name} in: {exc}"
                        )
                    ) from exc
                self.tensor_choices[tensor] = normalized
                if len(normalized) == 1:
                    self.tensor_fixed.append(tensor)
                else:
                    self.tensor_var.append(tensor)

        for tr in self.transfer_nodes:
            choices = _normalize_path_choices(self.mapping.get(tr).resource_allocation)
            ratio = self.partial_sum_ratio[tr] = (
                self.partial_sum_element_bits(tr.inputs[0]) / tr.inputs[0].operand_type.bitwidth
            )
            self.path_choices[tr] = tuple(with_partial_sum_shares(choice, ratio) for choice in choices)

    def _ensure_same_ssis_for_all_transfers(self) -> None:
        first = self.transfer_nodes[0]
        first_total = prod(self.ssis[first].get_temporal_sizes())
        for tr in self.transfer_nodes:
            total = prod(self.ssis[tr].get_temporal_sizes())
            if total != first_total:
                raise ValueError(
                    f"Transfer {tr.name} has different SSIS total size than the {first.name}: {total} != {first_total}"
                )

    def _init_transfer_fire_helpers(self) -> None:
        for t in self.workload.tensors:
            ssis = self.ssis[t].get_applicable_temporal_variables()
            sizes = [iter_var.size for iter_var in ssis]
            relevancies = [iter_var.relevant for iter_var in ssis]
            if any(iter_var.reuse != Reuse.NOT_SET for iter_var in ssis):
                continue
            self.tensors_to_optimize_reuse_for.append(t)
            reuse_factor = 1
            tiles_factor = 1
            self.reuse_levels[(t, -1)] = reuse_factor
            self.tiles_needed_levels[(t, -1)] = tiles_factor
            self.bds_needed_levels[(t, -1)] = tiles_factor
            for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies, strict=True)):
                reuse_factor *= Nl if not relevancy else 1
                tiles_factor *= Nl if relevancy else 1
                self.reuse_levels[(t, i)] = reuse_factor
                self.tiles_needed_levels[(t, i)] = tiles_factor
                moving = zip(relevancies[i + 1 :], sizes[i + 1 :], strict=True)
                self.rotation_levels[(t, i)] = any(relevant and n > 1 for relevant, n in moving)
                self.bds_needed_levels[(t, i)] = 4 if i == len(sizes) - 1 else tiles_factor
            if tiles_factor > 1:
                self.tiles_needed_levels[(t, -1)] = 2

    def _index_choice_metadata(self) -> None:
        for tr in self.transfer_nodes:
            for choice in self.path_choices[tr]:
                key = (tr, choice)
                self.links_in_choice[key] = set() if self.in_one_memory(choice) else set(choice.links_used)
                self.link_set.update(self.links_in_choice[key])
                self.choice_src_memories[key] = set(choice.sources)
                self.choice_dst_memories[key] = set(choice.targets)
                self.choice_has_empty_path[key] = not choice.links_used and not self.in_one_memory(choice)

    def is_fixed(self, t: Tensor) -> bool:
        return t in self._fixed

    def fixed_choice(self, t: Tensor) -> Placement:
        choices = self.tensor_choices[t]
        assert len(choices) == 1, f"Tensor {t.name} is not fixed."
        return choices[0]

    def route_name(self, tr: TransferNode, choice: MulticastPathPlan) -> str:
        """``tr``'s name with the index of ``choice`` among its routes, to name what the model builds per route."""
        return f"{tr.name}_choice_{self.path_choices[tr].index(choice)}"

    def candidate_memories(self, t: Tensor) -> set[Memory]:
        """Every memory one of ``t``'s placements uses; the set is shared, so callers do not change it."""
        if t not in self._candidates:
            self._candidates[t] = {memory for choice in self.tensor_choices[t] for memory in choice}
        return self._candidates[t]

    def runtime(self, n: ComputationNode) -> int:
        """Cycles ``n`` takes on the slowest core it may run on, before its active fraction (:meth:`core_runtime`)."""
        latencies = [self.core_runtime(n, c) for c in self.cost_lut.get_cores(n)]
        return ceil(max(latencies)) if latencies else 0

    def core_runtime(self, n: ComputationNode, core: Core) -> float:
        """Cycles a call of ``n`` takes on ``core``: its cost backend's, less the write up to the top memory of a sum
        kept in the output buffer (:attr:`sums_in_output_buffer`) on every call but the last, averaged over the
        calls."""
        skipped = sum(
            share.offload * (1 - 1 / self.reduction_trips(t))
            for t in n.outputs
            if (share := self.sums_in_output_buffer.get(t, {}).get(core)) is not None
        )
        return self.cost_lut.get_cost(n, core).latency_total - skipped

    def active_runtime(self, n: ComputationNode) -> float:
        """Cycles ``n`` takes per iteration: its runtime over the iterations it is not idle on an absent loop."""
        return get_active_latency(n, self.runtime(n), self.ssis)

    def runs_in_epilogue(self, n: ComputationNode) -> bool:
        """Whether ``n`` runs once in the run, after the last iteration: an epilogue node absent from every loop."""
        return n in self.epilogue_nodes and absent_trips(self.ssis[n]) == self.iterations

    def calls_before_first_read(self, producer: HasIterationSpace, reader: HasIterationSpace) -> int:
        """Calls of ``producer`` the first call of ``reader`` waits for: up to its last along each loop it varies on
        and the reader does not."""
        loops = zip(
            self.ssis[producer].get_temporal_variables(), self.ssis[reader].get_temporal_variables(), strict=True
        )
        waited, inner = 1, 1
        for p, r in loops:
            if p.effect is LoopEffect.VARYING and r.effect is not LoopEffect.VARYING:
                waited += (p.size - 1) * inner
            inner *= p.size if p.effect is LoopEffect.VARYING else 1
        return waited

    @cached_property
    def waiting_absent_nodes(self) -> frozenset[ComputationNode]:
        """The computation nodes absent from a fused loop that a computation node they read from or feed varies on."""
        pairs = [(node, reader) for node in self.ssc_nodes for reader, _ in self._reads_of(node)]
        return frozenset(n for a, b in pairs for n, other in ((a, b), (b, a)) if self._absent_where_varies(n, other))

    def _absent_where_varies(self, node: ComputationNode, other: ComputationNode) -> bool:
        """Whether ``node`` is absent from a loop ``other`` varies on."""
        loops = zip(self.ssis[node].get_temporal_variables(), self.ssis[other].get_temporal_variables(), strict=True)
        return any(a.effect is LoopEffect.ABSENT and b.effect is LoopEffect.VARYING for a, b in loops)

    def producing_node(self, tr: TransferNode) -> ComputationNode | None:
        """The computation node whose results ``tr`` moves, across the transfers before it; None for a group input."""
        node = next(iter(self.workload.predecessors(tr)), None)
        while node in self.path_choices:
            node = next(iter(self.workload.predecessors(node)), None)
        return node if node in self.ssc_nodes else None

    def epilogue_stops(self, tr: TransferNode, copy: Tensor) -> frozenset[int]:
        """The reuse stops at which ``copy`` of ``tr`` is kept along a loop in which the node whose results it moves
        still computes, so that its last firing comes after that node's last call."""
        producer = self.producing_node(tr)
        if producer is None:
            return frozenset()
        loops = zip(self.ssis[copy].get_temporal_variables(), self.ssis[producer].get_temporal_variables(), strict=True)
        held = [not c.relevant and c.size > 1 and p.effect is LoopEffect.VARYING for c, p in loops if c.applicable]
        first = next((i for i, waits in enumerate(held) if waits), len(held))
        return frozenset(range(first, len(held)))

    @cached_property
    def epilogue_nodes(self) -> tuple[HasIterationSpace, ...]:
        """The transfers with a copy that has :meth:`epilogue_stops` and every node after them, in dataflow order."""
        nodes = {*self.ssc_nodes, *self.transfer_nodes}
        frontier = [tr for tr in self.transfer_nodes if any(self.epilogue_stops(tr, copy) for copy in tr.outputs)]
        found = set(frontier)
        while frontier:
            for n in self.workload.successors(frontier.pop()):
                if n in nodes and n not in found:
                    found.add(n)
                    frontier.append(n)
        return tuple(n for n in self.workload.topological_sort() if n in found)

    def once_per_run_stops(self, tr: TransferNode, copy: Tensor) -> frozenset[int]:
        """The reuse stops at which ``copy`` of ``tr`` fires once in the run, after the last iteration: a
        :meth:`epilogue_stops` stop, or any where ``tr`` moves the results of a node that
        :meth:`runs_in_epilogue`."""
        producer = next(iter(self.workload.predecessors(tr)), None)
        after = self.stops(copy) if producer in self.ssc_nodes and self.runs_in_epilogue(producer) else ()
        trips = absent_trips(self.ssis[tr])
        stops = {*self.epilogue_stops(tr, copy), *after}
        return frozenset(s for s in stops if trips * self.reuse_levels[(copy, s)] == self.iterations)

    def stops(self, t: Tensor) -> range:
        """The reuse stops ``t`` may take: -1 (none) up to its outermost applicable temporal loop."""
        return range(-1, len(self.ssis[t].get_applicable_temporal_variables()))

    def _reader_placements(self, tr: TransferNode, t: Tensor, choices: tuple[Placement, ...]) -> tuple[Placement, ...]:
        """Narrow a transfer's placements to the positions its reading computation node reads the copy at: in each
        memory, as many as the reader's cores read it there. A copy read by another transfer or an edge, or one no
        placement lands in those memories, keeps all placements."""
        reader = next(n for n in self.workload.successors(tr) if t in n.inputs)
        if not isinstance(reader, ComputationNode):
            return choices
        most: Counter[Memory] = Counter()
        for choice in self.placements(reader, t):
            most |= Counter(choice)
        narrowed = (tuple(m for i, m in enumerate(choice) if choice[:i].count(m) < most[m]) for choice in choices)
        return tuple(dict.fromkeys(choice for choice in narrowed if choice)) or choices

    def placements(self, node: Node, tensor: Tensor) -> tuple[Placement, ...]:
        """The memories ``node`` may hold or read ``tensor`` in, see :func:`placements_of`."""
        role = role_of(node, tensor) if isinstance(node, ComputationNode) else OUTPUT
        return placements_of(self.accelerator, self.mapping, node, role)

    def cores(self, node: ComputationNode) -> tuple[Core, ...]:
        """The cores ``node`` runs on."""
        return the_one(self.mapping.get(node).resource_allocation, node, "core allocations")

    def compute_cores(self, tr: TransferNode) -> frozenset[Core]:
        """The cores of the computation nodes that produce ``tr``'s input or read its outputs directly."""
        if tr not in self._compute_cores:
            nodes = (*self.workload.predecessors(tr), *self.workload.successors(tr))
            self._compute_cores[tr] = frozenset(
                core for node in nodes if isinstance(node, ComputationNode) for core in self.cores(node)
            )
        return self._compute_cores[tr]

    @cached_property
    def producers(self) -> dict[Tensor, ComputationNode]:
        """Each tensor a computation node writes, with that node."""
        return {t: node for node in self.ssc_nodes for t in node.outputs}

    def _cuts(self, t: Tensor) -> tuple[set[LayerDim], set[LayerDim]]:
        """The dims of ``t`` its fused loops cut, and those its producer's own split over cores cuts."""
        fused: set[LayerDim] = set()
        split: set[LayerDim] = set()
        for v in self.ssis[t].variables:
            if v.size > 1 and v.type in (IterationVariableType.TEMPORAL, IterationVariableType.SPATIAL):
                (fused if v.type is IterationVariableType.TEMPORAL else split).add(v.dimension)
        return fused, split

    def _top_memory_cuts(self, t: Tensor) -> tuple[set[LayerDim], set[LayerDim]]:
        """:meth:`_cuts`, without the fused loops for a sum kept in the output buffer (:attr:`sums_in_output_buffer`):
        the top memory holds it complete."""
        fused, split = self._cuts(t)
        return (set() if t in self.sums_in_output_buffer else fused), split

    @cached_property
    def sums_in_output_buffer(self) -> dict[Tensor, dict[Core, OutputBufferUse]]:
        """The running sums their producer's cores keep across fused loops in their output buffer, with that buffer's
        use on each core (:func:`kept_in_output_buffer`)."""
        calls: dict[Tensor, RunningSumCall] = {}
        for node in self.ssc_nodes:
            cores = self.cores(node)
            mappings = {c: self.cost_lut.get_cost(node, c).cme for c in cores if self.cost_lut.has_cost(node, c)}
            for t in node.outputs:
                elements = self.workload.get_tensor_single_core(t, node, self.mapping).size_elements()
                calls[t] = RunningSumCall(node, cores, elements, self._running_sum_tiles(t), mappings)
        return kept_in_output_buffer(calls)

    def _running_sum_tiles(self, t: Tensor) -> int:
        """Tiles of ``t`` its producer sums into at once: one per trip of the fused loops moving ``t`` inside the
        outermost one over a dim the producer reduces; 0 where no fused loop does."""
        reduced = reduced_dims(self.workload, self.producers[t])
        loops = self.ssis[t].get_temporal_variables()
        outer = max((i for i, v in enumerate(loops) if v.dimension in reduced and v.size > 1), default=None)
        return 0 if outer is None else prod(v.size for v in loops[:outer] if v.relevant)

    def running_sum_bits(self, t: Tensor) -> int | None:
        """Bits per element ``t`` is held in while it is a running sum: the accumulator's across a fused loop cutting
        a dimension the producer reduces over, unless the output buffer keeps it (:attr:`sums_in_output_buffer`), the
        partial-sum format where only its split over cores does. None otherwise."""
        if (node := self.producers.get(t)) is None:
            return None
        return running_sum_bits(
            self.workload, node, self.cores(node), *self._top_memory_cuts(t), self._accumulator(node)
        )

    def partial_sum_element_bits(self, t: Tensor) -> int:
        """Bits per element the cores of ``t``'s producer send each other their partial sums of ``t`` in: the
        partial-sum format they declare where wider than ``t``'s type, else its type."""
        if (node := self.producers.get(t)) is None:
            return t.operand_type.bitwidth
        _, split = self._cuts(t)
        found = partial_sum_format(self.cores(node)) if split & reduced_dims(self.workload, node) else None
        return max(ELEMENT_BITS[found] if found else 0, t.operand_type.bitwidth)

    def sums_on_one_core(self, t: Tensor) -> bool:
        """Whether no split over cores cuts a dimension the producer of ``t`` reduces over."""
        reduced = reduced_dims(self.workload, self.producers[t])
        split = {v.dimension for v in self.ssis[t].variables if v.type is IterationVariableType.SPATIAL and v.size > 1}
        return not split & reduced

    def reduction_trips(self, t: Tensor) -> int:
        """The trips of the fused loops over the dimensions the producer of ``t`` reduces."""
        reduced = reduced_dims(self.workload, self.producers[t])
        loops = self.ssis[t].get_temporal_variables()
        return prod(v.size for v in loops if v.dimension in reduced)

    def running_sum_format(self, t: Tensor) -> str | None:
        """The format ``t`` is held in while it is a running sum, else None."""
        if (node := self.producers.get(t)) is None:
            return None
        return running_sum_format(
            self.workload, node, self.cores(node), *self._top_memory_cuts(t), self._accumulator(node)
        )

    def _accumulator(self, node: ComputationNode) -> str | None:
        """The format ``node``'s kernel keeps its running sum in, where it declares one."""
        return getattr(self.mapping.get(node).kernel, "ACCUMULATOR", None)

    def element_bits(self, t: Tensor) -> int:
        """Bits per element ``t`` is held at: a running sum's width (:meth:`running_sum_bits`), else its type's."""
        return self.running_sum_bits(t) or t.operand_type.bitwidth

    def moved_element_bits(self, tr: TransferNode, choice: MulticastPathPlan) -> int:
        """Bits per element ``tr`` moves at on ``choice``: the partial-sum format of its source's cores
        (:meth:`partial_sum_element_bits`) where it carries them to be added up on chip, else its type's."""
        tensor = tr.inputs[0]
        hops = zip(choice.hops, choice.hop_demand, strict=True)
        partial = any(item.reduce for (_, v, _), demand in hops if not v.offchip for item in demand)
        return self.partial_sum_element_bits(tensor) if partial else tensor.operand_type.bitwidth

    def held_bits(self, t: Tensor, tile: Tensor | None = None) -> int:
        """Bits of ``tile`` (default all) of ``t`` held, at :meth:`element_bits`, or at its type where its kernel sums
        into an accumulator of its own (:meth:`accumulator_bits`)."""
        bits = t.operand_type.bitwidth if self.accumulator_bits(t) else self.element_bits(t)
        return (t if tile is None else tile).size_elements() * bits

    def accumulator_bits(self, t: Tensor, tile: Tensor | None = None) -> int:
        """Bits of the buffer the kernel producing ``t`` declares it sums ``tile`` (default all) of it into, held beside
        the tiles of ``t`` it sends on, one per tile it sums at once (:meth:`summed_tiles`); 0 where its kernel declares
        none."""
        node = self.producers.get(t)
        if node is None or (accumulator := self._accumulator(node)) is None:
            return 0
        return (t if tile is None else tile).size_elements() * ELEMENT_BITS[accumulator]

    @cached_property
    def summed_outputs(self) -> frozenset[Tensor]:
        """The outputs whose kernel sums them in accumulators of its own and sends each tile on once its sum is done,
        through one buffer, while the next sum runs in the accumulators."""
        return frozenset(t for node in self.ssc_nodes for t in node.outputs if self.accumulator_bits(t))

    def summed_tiles(self, t: Tensor, stop: int) -> int:
        """Tiles of ``t`` its producer sums at once held to ``stop``: those its window spans, one where it holds a
        single tile."""
        return self.tiles_needed_levels[(t, stop)] if stop >= 0 else 1

    @cached_property
    def kernel_states(self) -> frozenset[Tensor]:
        """The recurrence states the computation nodes carry, each a buffer of its kernel's own."""
        return frozenset(t for node in self.ssc_nodes for t in node.inputs if is_state_operand(node, t))

    @cached_property
    def kernel_scratch(self) -> dict[Memory, int]:
        """Per memory, the bits of the scratch buffer the kernels running on its core keep there (``SCRATCH_BITS``),
        one buffer a core whichever of its kernels makes it."""
        found: dict[Memory, int] = {}
        for node in self.ssc_nodes:
            bits = getattr(self.mapping.get(node).kernel, "SCRATCH_BITS", 0)
            for memory in the_one(self.placements(node, node.outputs[0]), node, "placements") if bits else ():
                found[memory] = max(found.get(memory, 0), bits)
        return found

    def partial_sum_factor(
        self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, leaving: bool, copies: CopySet | None = None
    ) -> float:
        """The factor on the bits ``memory`` sends (``leaving``) or receives of ``tr`` on ``choice`` for the partial
        sums among them, moved at the partial-sum format."""
        if (ratio := self.partial_sum_ratio.get(tr, 1.0)) == 1:
            return 1.0
        hops = _hop_elements(choice, ratio, copies)
        own = [(n, w) for (u, v, _), n, w in hops if (u if leaving else v) is memory]
        elements = sum(n for n, _ in own)
        return sum(w for _, w in own) / elements if elements else 1.0

    def memory_share(
        self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, leaving: bool, copies: CopySet | None = None
    ) -> float:
        """The share of the delivered bits of ``tr``, or of those delivered to exactly ``copies``, ``memory`` sends or
        receives on ``choice``, its partial sums at the partial-sum format, with the reads and writes of the partial
        sums added into it."""
        share = choice.memory_share(memory, leaving, copies) * self.partial_sum_factor(
            tr, choice, memory, leaving, copies
        )
        if memory not in choice.sum_memories():
            return share
        links = (*choice.links_at(memory, leaving=True), *choice.links_at(memory, leaving=False))
        share += choice.local_sum_share(memory, links, copies) * self.partial_sum_ratio.get(tr, 1.0)
        if not leaving:
            return share
        added = choice.sum_read_share(memory, choice.links_at(memory, leaving=False), copies)
        return share + added * self.partial_sum_factor(tr, choice, memory, False, copies)

    @cached_property
    def sole_outputs(self) -> set[Tensor]:
        """The tensors a transfer moves to a single reader."""
        return {tr.outputs[0] for tr in self.transfer_nodes if len(set(self.workload.successors(tr))) == 1}

    def may_single_buffer(self, t: Tensor, stop: int) -> bool:
        """Whether a tensor a transfer moves to one reader may hold its window at ``stop`` in a single
        buffer where an outer loop moves it on, a one-tile window or one a memory tile holds whole:
        the next window then waits for the last read of this one."""
        return (
            stop >= 0
            and t in self.tensors_to_optimize_reuse_for
            and t in self.sole_outputs
            and self.rotation_levels[(t, stop)]
            and (self.tiles_needed_levels[(t, stop)] == 1 or self.staged_whole(t))
        )

    def staged_whole(self, t: Tensor) -> bool:
        """Whether ``t`` sits only in memory tiles, which hold each window as one object of all its tiles."""
        return all(memory.kind == "memory" for memory in self.candidate_memories(t))

    def staging_block(self, t: Tensor) -> tuple[int, dict[LayerDim, int]]:
        """The block a memory tile stages of ``t``: its innermost loops that tile a dim within a larger tile of the
        same dim, as the level up to which it holds ``t`` and their trips per dim; ``(-1, {})`` for none."""
        if not self.staged_whole(t):
            return -1, {}
        loops = self.ssis[t].get_applicable_temporal_variables()
        level, trips = -1, {}
        for i, loop in enumerate(loops):
            if not any(outer.dimension == loop.dimension for outer in loops[i + 1 :]):
                break
            level, trips[loop.dimension] = i, trips.get(loop.dimension, 1) * loop.size
        return level, trips

    def resident_tiles(self, t: Tensor, stop: int, single: bool = False, staged: bool = False) -> int:
        """Tiles of this tensor codegen keeps resident when it stops reuse at ``stop``, held in one
        buffer where ``single``; one of a kernel's state (:attr:`kernel_states`). A memory tile
        (``staged``) holds the window as one block, so one it moves on takes two of them."""
        if t in self.kernel_states:
            return 1
        if t in self.summed_outputs:
            return 1
        tiles, double_buffered = (
            self.tiles_needed_levels[(t, stop)],
            bool(self.rotation_levels.get((t, stop))) and not single,
        )
        return 2 * tiles if staged and double_buffered else buffer_count(tiles, double_buffered)

    def halo_bits(self, t: Tensor, stop: int) -> int:
        """Bits of halo the copy ``t`` holds on one core in addition to its resident tiles at reuse stop ``stop``."""
        return self._kept_halos.get(t, {}).get(stop, 0) * self.element_bits(t)

    @cached_property
    def _kept_halos(self) -> dict[Tensor, dict[int, int]]:
        """:meth:`Workload.kept_halos` of each transfer's copy that has any."""
        return {
            copy: kept
            for tr in self.transfer_nodes
            for copy in tr.outputs
            if (kept := self.workload.kept_halos(copy, tr, self.mapping, self.ssis.get(copy)))
        }

    def memory_capacity_bits(self, memory: Memory) -> int:
        """Bits of ``memory`` left for tensors: its capacity less what the toolchain claims in it."""
        return self.hardware.usable_bits(memory)

    def is_const_i(self, tr: TransferNode) -> bool:
        return is_const_input(self.workload, tr)

    def is_const_o(self, tr: TransferNode) -> bool:
        return is_const_output(self.workload, tr)

    def is_const_io(self, tr: TransferNode) -> bool:
        return self.is_const_i(tr) or self.is_const_o(tr)

    def constant_transfer_tensor(self, tr: TransferNode) -> Tensor:
        if self.is_const_i(tr):
            return tr.outputs[0]
        if self.is_const_o(tr):
            return tr.inputs[0]
        raise ValueError(f"Transfer {tr.name} is not a constant I/O transfer.")

    def transfer_latency_for_path(
        self, tr: TransferNode, path: MulticastPathPlan, copies: CopySet | None = None
    ) -> int:
        """Cycles one firing of ``tr`` takes on ``path``, before its active fraction and reuse; with ``copies``, the
        cycles of what it delivers to exactly those copies."""
        key = (tr, path, copies)
        if key not in self._firing_cycles:
            self._firing_cycles[key] = self._cycles_of_firing(tr, path, copies)
        return self._firing_cycles[key]

    def _cycles_of_firing(self, tr: TransferNode, path: MulticastPathPlan, copies: CopySet | None) -> int:
        """:meth:`transfer_latency_for_path`, computed."""
        if self.choice_shares_memory(tr, path):
            return 0
        if copies is None:
            link = get_transfer_latency_for_path(tr, path, self.moved_bits(tr, path))
        else:
            link = self._copy_set_link_cycles(tr, path, copies)
        shared = (
            self.shared_cycles(memory, tr, path, model.contiguous, copies)
            for memory, model in self.shared_bandwidth.items()
        )
        return max(link, *shared) if self.shared_bandwidth else link

    def core_stream_cycles(self, tr: TransferNode, path: MulticastPathPlan) -> float:
        """Cycles one firing of ``tr`` takes to reach the compute target that receives most of it, each at the width of
        the links of ``path`` into it: what a core's own stream needs, whatever else shares the route's links."""
        cycles = 0.0
        for at, target in enumerate(path.targets):
            links = path.links_at(target, leaving=False) if target.kind == "compute" else ()
            if links:
                width = sum(link.bandwidth for link in links)
                cycles = max(cycles, self.moved_bits(tr, path, target_at=at) / width)
        return cycles

    def _copy_set_link_cycles(self, tr: TransferNode, path: MulticastPathPlan, copies: CopySet) -> int:
        """Cycles the busiest link of ``path`` takes to carry what it delivers to exactly ``copies``."""
        bits = self.copy_set_link_bits(tr, path, copies)
        return ceil(max((b / link.bandwidth for link, b in bits.items()), default=0))

    def copy_set_link_bits(
        self, tr: TransferNode, path: MulticastPathPlan, copies: CopySet
    ) -> dict[CommunicationLink, float]:
        """Per link of ``path``, the bits one firing carries over it of what it delivers to exactly ``copies``, partial
        sums at the partial-sum format."""
        ratio = self.partial_sum_ratio.get(tr, 1.0)
        carried: dict[CommunicationLink, dict[tuple[int | None, bool], list[Box]]] = {}
        for (_, v, link), demand in zip(path.hops, path.hop_demand, strict=True):
            for item in demand:
                if item.copies == copies:
                    key = (None if item.reduce else item.sources[0], ratio != 1 and item.reduce and not v.offchip)
                    carried.setdefault(link, {}).setdefault(key, []).append(item.box)
        bitwidth = tr.inputs[0].operand_type.bitwidth / firing_count(path.demand)
        return {
            link: sum(box_bits(boxes, bitwidth * (ratio if partial else 1.0)) for (_, partial), boxes in sent.items())
            for link, sent in carried.items()
        }

    def link_shares(
        self, tr: TransferNode, path: MulticastPathPlan, copies: CopySet | None = None
    ) -> tuple[tuple[CommunicationLink, float], ...]:
        """Each link of ``path`` with the share of the bits it delivers, or of those it delivers to exactly
        ``copies``, that crosses it."""
        if copies is None:
            return path.link_shares
        moved = self.moved_bits(tr, path, copies=copies)
        return tuple((link, b / moved) for link, b in self.copy_set_link_bits(tr, path, copies).items() if moved and b)

    @staticmethod
    def direction(memory: Memory, path: MulticastPathPlan) -> str | None:
        """'read' for a transfer out of this memory, 'write' for one into it, None for one it takes no part in."""
        if memory in path.sources:
            return "read"
        if memory in path.targets:
            return "write"
        return None

    def shared_cycles(
        self, memory: Memory, tr: TransferNode, path: MulticastPathPlan, rate: float, copies: CopySet | None = None
    ) -> int:
        """Cycles one firing holds a shared-bandwidth memory at ``rate`` for its share of the bits, slowed by its
        access pattern."""
        direction = self.direction(memory, path)
        if direction is None:
            return 0
        span = self.layouts.runs(tr, path)[0 if direction == READ else 1]
        adjacent = self.layouts.adjacent_runs(tr, path, memory, direction)
        bits = self.moved_bits(tr, path, copies=copies) * self.memory_share(tr, path, memory, direction == READ, copies)
        return ceil(bits / (rate * self.shared_bandwidth[memory].efficiency(span, direction, adjacent)))

    @cached_property
    def layouts(self) -> CopyLayouts:
        """The layout of every tensor copy, and the contiguous runs each transfer moves them in."""
        return CopyLayouts(self)

    def side_rate(self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, direction: str) -> Rate:
        """Bits per cycle ``memory`` moves for one side of ``tr`` on route ``choice``, as a function of the contiguous
        run length. Unbounded when nothing limits it (see :meth:`side_bandwidth`)."""
        model = self.side_bandwidth(tr, choice, memory, direction)
        if model is None:
            return lambda span: float("inf")
        return lambda span: model.contiguous * model.efficiency(span, direction)

    def side_ports(
        self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, direction: str, copies: CopySet | None = None
    ) -> dict[Port, float]:
        """The ports of ``memory`` that ``tr``'s operand passes in ``direction`` on route ``choice``, each with the
        share of the transfer's bits it carries (:meth:`MulticastPathPlan.share_at`), with the reads and writes of the
        partial sums added into it. Empty when the route moves no bits through a port of ``memory``. A target passing
        data on reads it in the role it holds it in. Callers do not change the dict, which is shared."""
        key = (tr, choice, memory, direction, copies)
        if key not in self._side_ports:
            self._side_ports[key] = self._ports_of_side(tr, choice, memory, direction, copies)
        return self._side_ports[key]

    def _ports_of_side(
        self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, direction: str, copies: CopySet | None
    ) -> dict[Port, float]:
        """:meth:`side_ports`, computed."""
        leaving = direction == READ
        role = self.operand_role(tr, read_side=leaving and memory in choice.sources)
        by_port = self.accelerator.ports.memory_ports(memory, direction, role, choice.links_at(memory, leaving))
        widening = self.partial_sum_factor(tr, choice, memory, leaving, copies)
        shares = {
            port: choice.share_at(memory, leaving, port_links, copies) * widening
            for port, port_links in by_port.items()
        }
        if memory in choice.sum_memories():
            ratio = self.partial_sum_ratio.get(tr, 1.0)
            links = (*choice.links_at(memory, leaving=True), *choice.links_at(memory, leaving=False))
            for port, port_links in self.accelerator.ports.memory_ports(memory, direction, role, links).items():
                shares[port] = shares.get(port, 0.0) + choice.local_sum_share(memory, port_links, copies) * ratio
            arriving = choice.links_at(memory, leaving=False) if leaving else ()
            received = self.partial_sum_factor(tr, choice, memory, False, copies)
            for port, port_links in self.accelerator.ports.memory_ports(memory, READ, role, arriving).items():
                shares[port] = shares.get(port, 0.0) + choice.sum_read_share(memory, port_links, copies) * received
        return {port: share for port, share in shares.items() if share}

    def side_port(self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, direction: str) -> Port | None:
        """Of :meth:`side_ports`, the one carrying the most of the side's bits; None where none is."""
        ports = self.side_ports(tr, choice, memory, direction)
        return max(ports, key=ports.__getitem__, default=None)

    def side_bandwidth(
        self, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, direction: str
    ) -> BandwidthModel | None:
        """What bounds ``memory``'s side of ``tr`` on route ``choice``: its measured shared bandwidth over its share of
        the bits, else the port needing longest for its share (ports work in parallel), else None."""
        if memory in self.shared_bandwidth:
            share = self.memory_share(tr, choice, memory, leaving=direction == READ)
            return self.shared_bandwidth[memory].scaled(1.0 / share) if share else None
        ports = self.side_ports(tr, choice, memory, direction)
        if not ports:
            return None
        port = min(ports, key=lambda port: port.bandwidth.contiguous / ports[port])
        return port.bandwidth.scaled(1.0 / ports[port])

    def walk_shares(self, tr: TransferNode, choice: MulticastPathPlan) -> dict[Memory, float]:
        """Each memory's share of what ``tr`` moves of an edge whose placement splits it, an input as ``choice`` holds
        it, over all iterations walking the split axis (:meth:`Workload.placement_shares`); empty for any other
        transfer."""
        held = [choice.placement or self.mapping.placement_of(edge) for edge in self.workload.predecessors(tr)]
        landed = [self.mapping.placement_of(edge) for edge in self.workload.successors(tr)]
        for placement in (*held, *landed):
            if (shares := self.workload.placement_shares(placement, tr.inputs[0], choice.delivered)) is not None:
                return shares
        return {}

    def memory_links(self, memory: Memory, leaving: bool) -> tuple[CommunicationLink, ...]:
        """The links leaving (or entering) ``memory``."""
        edges = self.accelerator.network.out_edges if leaving else self.accelerator.network.in_edges
        return tuple(dict.fromkeys(data["cl"] for *_, data in edges(memory, data=True)))

    def operand_role(self, tr: TransferNode, read_side: bool) -> str:
        """Operand role the tensor has on one side: the output for a producer, input k for a consumer's input k."""
        producer = next(iter(self.workload.predecessors(tr)), None)
        if read_side and isinstance(producer, ComputationNode):
            return OUTPUT
        for consumer in self.workload.successors(tr):
            if isinstance(consumer, ComputationNode):
                for tensor in tr.outputs:
                    if tensor in consumer.inputs:
                        return role_of(consumer, tensor)
        return OUTPUT

    def offchip_bandwidth(self) -> float:
        """Bits per cycle the array can move across the off-chip boundary."""
        offchip = self.accelerator.offchip_memories
        return float(
            sum(
                link.bandwidth
                for link in self.link_set
                if link.receiver in offchip or any(memory in link.members for memory in offchip)
            )
        )

    def placement_width(self, tensors: Iterable[Tensor]) -> int:
        """How many positions one side of a transfer occupies."""
        return max((len(choice) for t in tensors for choice in self.tensor_choices[t]), default=1)

    def distinct_slice_width(self, tensors: Iterable[Tensor]) -> int:
        """How many distinct slices one side of a transfer holds: cores a spatial loop does not address separately
        read one slice, which the object-fifo lowering serves from one channel."""
        return max(
            (
                prod(v.size for v in self.ssis[t].variables if v.type is IterationVariableType.SPATIAL and v.relevant)
                for t in tensors
                if t in self.ssis
            ),
            default=1,
        )

    def transfer_is_broadcast(self, tr: TransferNode) -> bool:
        """Whether several cores are served the same slice, so one fifo carries them all, on the DMA however the
        cores are placed (``requiresDMAs`` only looks at the tiles of a fifo with one consumer)."""
        if (broadcast := self._broadcast.get(tr)) is None:
            tensors = unique_tensors(tr.outputs)
            broadcast = self._broadcast[tr] = self.distinct_slice_width(tensors) < self.placement_width(tensors)
        return broadcast

    def transfer_shares_memory(self, tr: TransferNode, memory: Memory, incoming: bool) -> bool:
        """Whether this memory gets (or gives) this transfer's data without moving it: every memory it takes from (or
        serves) shares memory with it."""
        if self.within_one_memory(tr):
            return True
        choices = self.path_choices.get(tr) or ()
        if not choices or self.transfer_is_broadcast(tr):
            return False
        if tr.transfer_type is not TransferType.COMPUTE_TO_COMPUTE:
            return False
        for choice in choices:
            touching = [(a, b) for a, b in self.pairs(tr, choice) if (b if incoming else a) is memory]
            if not touching:
                return False
            if any(not self.hardware.shares_memory(one, other) for one, other in touching):
                return False
        return True

    def choice_shares_memory(self, tr: TransferNode, choice: MulticastPathPlan) -> bool:
        """Whether this transfer, placed on this choice, lands core to core out of memory the two
        sides already share -- the object-fifo lowering that spends no DMA channel and moves no bytes
        over a link. The per-choice form of the same conditions ``transfer_shares_memory`` reads."""
        if self.in_one_memory(choice):
            return True
        if tr.transfer_type is not TransferType.COMPUTE_TO_COMPUTE or self.transfer_is_broadcast(tr):
            return False
        pairs = self.pairs(tr, choice)
        return bool(pairs) and all(self.hardware.shares_memory(one, other) for one, other in pairs)

    def overlaps(self, tr: TransferNode) -> dict[tuple[int, int], float]:
        """Elements per firing each source position serves each target position where windows overlap tiles."""
        if (found := self._overlaps.get(tr)) is None:
            found = self._overlaps[tr] = self.workload.get_transfer_overlaps(tr, self.mapping, self.ssis[tr.outputs[0]])
        return found

    def moved_bits(
        self,
        tr: TransferNode,
        choice: MulticastPathPlan,
        source_at: int | None = None,
        target_at: int | None = None,
        copies: CopySet | None = None,
    ) -> float:
        """Bits one firing of ``tr`` moves on ``choice``, each source's data once however many targets and copies read
        it. With ``source_at``, ``target_at`` or ``copies``, only what leaves the source at that position, enters the
        target at that position, or reaches exactly those copies."""
        items = [
            item
            for item in choice.delivered_items
            if (source_at is None or source_at in item.sources)
            and target_at in (None, item.target)
            and copies in (None, item.copies)
        ]
        return moved_elements(items) * tr.inputs[0].operand_type.bitwidth

    def sent_bits(self, tr: TransferNode, choice: MulticastPathPlan, copies: CopySet | None = None) -> float:
        """:meth:`moved_bits` of ``choice``, or of what it delivers to exactly ``copies``, with the partial sums added
        up on chip at their :meth:`partial_sum_element_bits`."""
        items = [item for item in choice.delivered_items if copies in (None, item.copies)]
        ratio = self.partial_sum_ratio.get(tr, 1.0)
        partial = [item for item in items if item.reduce and not choice.targets[item.target].offchip]
        extra = (ratio - 1) * moved_elements(partial) * firing_count(partial) / firing_count(items) if partial else 0.0
        return (moved_elements(items) + extra) * tr.inputs[0].operand_type.bitwidth

    def copy_sets(self, tr: TransferNode, choice: MulticastPathPlan) -> tuple[CopySet, ...]:
        """The copy sets ``choice`` delivers some box to: each a copy alone, or several reading the same box."""
        return copy_sets(choice.delivered_items)

    def copied_bits(self, t: Tensor) -> float:
        """Bits per firing of the demand of ``t`` on the transfer writing it, moved or in place, each element once; for
        a tensor no transfer writes, the demand of all copies of the transfer reading it."""
        landing = next((tr for tr in self.transfer_nodes if t in tr.outputs), None)
        tr = landing or next((tr for tr in self.transfer_nodes if t in tr.inputs), None)
        if tr is None:
            return t.size_bits()
        items: list[DemandItem] = list(self.path_choices[tr][0].demand)
        if landing is not None:
            items = [item for item in items if tr.outputs.index(t) in item.copies]
        return item_bits(items, tr.inputs[0].operand_type.bitwidth)

    def pairs(self, tr: TransferNode, choice: MulticastPathPlan) -> tuple[tuple[Memory, Memory], ...]:
        """The sources and targets of ``choice`` that serve ``tr``'s data to each other: those its route pairs up."""
        if choice.pairs:
            return tuple((choice.sources[i], choice.targets[j]) for i, j in choice.pairs)
        return communicating_pairs(choice.sources, choice.targets, self.overlaps(tr))

    def link_load(
        self, tr: TransferNode, choice: MulticastPathPlan, copies: CopySet | None = None
    ) -> dict[CommunicationLink, float]:
        """Per link of ``choice``, the share of one firing's latency it is busy: the bottleneck link 1."""
        if not (links := self.links_in_choice[(tr, choice)]):
            return {}
        if not choice.link_shares:
            return dict.fromkeys(links, 1.0)
        shares = self.link_shares(tr, choice, copies)
        cycles = {link: share / link.bandwidth for link, share in shares if link in links and share}
        slowest = max(cycles.values(), default=0.0)
        return {link: c / slowest for link, c in cycles.items()}

    def in_one_memory(self, choice: MulticastPathPlan) -> bool:
        """Whether each source of this choice serves its targets data already in their memory, so none of it moves:
        every pair its route makes is one memory, or, without pairs, all of it is."""
        if choice.pairs:
            return all(choice.sources[i] is choice.targets[j] for i, j in choice.pairs)
        return len({*choice.sources, *choice.targets}) == 1

    def within_one_memory(self, tr: TransferNode) -> bool:
        """Whether every placement of this transfer stays in one memory."""
        if (within := self._one_memory.get(tr)) is None:
            choices = self.path_choices.get(tr)
            within = self._one_memory[tr] = bool(choices) and all(self.in_one_memory(choice) for choice in choices)
        return within

    @cached_property
    def handovers(self) -> tuple[tuple[Memory, Memory, int], ...]:
        """Kernel state (such as a running reduction's scale) passed to the core consuming the kernel's output, as
        ``(memory of the passing core, memory the consumer reads in, bits)``. Both ends hold a buffer of that size;
        which cores pair up follows the transfer between them."""
        found: list[tuple[Memory, Memory, int]] = []
        for node in self.ssc_nodes:
            kernel = self.mapping.get(node).kernel
            for state in kernel.state_operands() if kernel else ():
                if not state.handover:
                    continue
                held = next((t for t in node.inputs if is_state_operand(node, t)), None)
                if held is None:
                    continue
                bits = self.workload.get_tensor_single_core(held, node, self.mapping).size_bits()
                sources = the_one(self.placements(node, held), node, "placements")
                for consumer, read in self._reads_of(node, chained=True):
                    targets = the_one(self.placements(consumer, read), consumer, "placements")
                    pairs = communicating_pairs(sources, targets)
                    found += [(one, other, state.handover * bits) for one, other in pairs]
        return tuple(found)

    @cached_property
    def warmup(self) -> dict[HasIterationSpace, float]:
        """How far, in interior tiles, each node's work along the sliding windows runs ahead of its steady pace at most,
        the fused loops nested as listed, innermost first: the lookahead a producer computes and the halo a transfer
        moves before a window slides, each pass over an inner loop's trips restarting them; along a dim its split
        places, a producer runs no lookahead but its readers wait for it (:meth:`neighbour_waits`)."""
        found: dict[HasIterationSpace, float] = {}
        if not any(loop.halo for ssis in self.ssis.values() for loop in ssis):
            return found
        splits = self.problem.fusion_splits
        rates: dict[HasIterationSpace, list] = {}
        for k, (dim, n) in enumerate(splits.items()):
            for node, work in self.workload.get_sliding_work(dim, n).items():
                if dim not in self._split_dims(node):
                    rates.setdefault(node, [np.ones(m) for m in splits.values()])[k] = np.array(work) * n / sum(work)
        for node, rate in rates.items():
            excess = np.cumsum(reduce(np.multiply.outer, reversed(rate)).ravel() - 1).max()
            if excess > 0:
                found[node] = float(excess)
        for node, lead in self.neighbour_waits.items():
            found[node] = found.get(node, 0.0) + lead
        return found

    @cached_property
    def neighbour_waits(self) -> dict[ComputationNode, int]:
        """Per producer, the most firings ahead of its own a reader on another core reads its tiles at
        (:meth:`Workload.neighbour_wait`)."""
        found: dict[ComputationNode, int] = {}
        for tr in self.transfer_nodes:
            src = next(iter(self.workload.predecessors(tr)))
            if isinstance(src, ComputationNode) and (
                lead := self.workload.neighbour_wait(tr, self.mapping, self.problem.fusion_splits)
            ):
                found[src] = max(found.get(src, 0), lead)
        return found

    def _split_dims(self, node: HasIterationSpace) -> set[LayerDim]:
        """The dims a computation node's split spreads over several cores; none for a transfer."""
        if not isinstance(node, ComputationNode):
            return set()
        return {d for d, f in self.workload.get_unique_dims_inter_core_tiling(node, self.mapping) if f > 1}

    def is_waited_on(self, node: ComputationNode) -> bool:
        """Whether a reader of ``node``'s output waits for its tiles: one on a core ``node`` does not run on, or behind
        a transfer that takes cycles on some route."""
        after = [tr for tr in self.workload.successors(node) if tr in self.path_choices]
        cores = set(self.cores(node))
        return any(self.transfer_latency_for_path(tr, c) for tr in after for c in self.path_choices[tr]) or any(
            not cores.issuperset(self.cores(c)) for c, _ in self._reads_of(node)
        )

    def _reads_of(
        self, node: ComputationNode | TransferNode, chained: bool = False
    ) -> list[tuple[ComputationNode, Tensor]]:
        """The computation nodes this one's output reaches, across the transfer between them, or with ``chained``
        across every copy of a chain of them, each with the tensor it reads it as."""
        reached = []
        for succ in self.workload.successors(node):
            if isinstance(succ, ComputationNode):
                reached.append((succ, next(t for t in succ.inputs if t in node.outputs)))
            elif isinstance(succ, TransferNode) and chained:
                reached += self._reads_of(succ, chained)
            elif isinstance(succ, TransferNode):
                readers = (x for x in self.workload.successors(succ) if isinstance(x, ComputationNode))
                reached += [(x, next(t for t in x.inputs if t in succ.outputs)) for x in readers]
        return reached


def is_const_input(workload: Workload, tr: TransferNode) -> bool:
    """Whether ``tr`` moves a tensor the workload takes in."""
    return isinstance(next(iter(workload.predecessors(tr))), InEdge)


def is_const_output(workload: Workload, tr: TransferNode) -> bool:
    """Whether ``tr`` moves a tensor the workload puts out."""
    return isinstance(next(iter(workload.successors(tr))), OutEdge)


def with_partial_sum_shares(choice: MulticastPathPlan, ratio: float) -> MulticastPathPlan:
    """``choice`` with each link's share of the delivered bits ``ratio`` times as large for the partial sums it carries
    to be added up on chip."""
    if ratio == 1 or not choice.link_shares:
        return choice
    carried: dict[CommunicationLink, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (_, _, link), elements, widened in _hop_elements(choice, ratio):
        carried[link][0] += elements
        carried[link][1] += widened
    shares = []
    for link, share in choice.link_shares:
        elements, widened = carried[link]
        shares.append((link, share * widened / elements if elements else share))
    return replace(choice, link_shares=tuple(shares))


def _hop_elements(
    choice: MulticastPathPlan, ratio: float, copies: CopySet | None = None
) -> Iterator[tuple[Hop, float, float]]:
    """Per hop of ``choice``, the elements it carries (to exactly ``copies``, if given), and those with its partial
    sums ``ratio`` times over, except on a hop into off-chip memory."""
    for hop, carried in zip(choice.hops, choice.hop_demand, strict=True):
        sent: dict[int | None, list[Box]] = {}
        for item in carried:
            if copies in (None, item.copies):
                sent.setdefault(None if item.reduce else item.sources[0], []).append(item.box)
        elements = sum(covered(boxes) for boxes in sent.values())
        partial = covered(sent.get(None, [])) if not hop[1].offchip else 0
        yield hop, elements, elements + (ratio - 1) * partial


def unique_tensors(tensors: Iterable[Any]) -> list[Tensor]:
    """The tensors among ``tensors``, each once, in order."""
    return list(dict.fromkeys(t for t in tensors if isinstance(t, Tensor)))


def communicating_pairs(
    src: Sequence[Memory], dst: Sequence[Memory], overlaps: Mapping[tuple[int, int], float] | None = None
) -> tuple[tuple[Memory, Memory], ...]:
    """Which ``src`` positions hand to which ``dst`` positions: those whose tiles ``overlaps`` the targets' windows,
    else by spatial index as codegen pairs them, its contract and why overlaps only pair windowed transfers: the
    target at ``j`` is fed by the sources at ``j``, ``j + m``, ..., ``m`` being the narrower side."""
    if not src or not dst:
        return ()
    if overlaps:
        return tuple((src[i], dst[j]) for i, j in overlaps)
    narrow = min(len(src), len(dst))
    return tuple((src[i], dst[j]) for i in range(len(src)) for j in range(len(dst)) if i % narrow == j % narrow)


def _normalize_tensor_choices(raw: Any) -> tuple[Placement, ...]:
    """The placements a node's allocation offers, as a tuple of memory tuples."""
    if raw is None:
        raise ValueError("Tensor allocation options cannot be None.")
    raw_tuple = tuple(raw)
    if not raw_tuple:
        raise ValueError("Tensor allocation options cannot be empty.")
    if all(isinstance(x, Memory) for x in raw_tuple):
        return (tuple(raw_tuple),)
    out: list[Placement] = []
    for choice in raw_tuple:
        choice_tuple = tuple(choice)
        if not choice_tuple:
            raise ValueError("Empty tensor placement choice encountered.")
        if not all(isinstance(c, Memory) for c in choice_tuple):
            raise TypeError(f"Invalid tensor placement choice: {choice_tuple}")
        out.append(choice_tuple)
    return tuple(out)


def _normalize_path_choices(raw: Any) -> tuple[MulticastPathPlan, ...]:
    """The routes a transfer's allocation offers, as a tuple of path plans."""
    if raw is None:
        raise ValueError("Transfer path options cannot be None.")
    raw_tuple = tuple(raw)
    if not raw_tuple:
        raise ValueError("Transfer path options cannot be empty.")
    if not all(isinstance(x, MulticastPathPlan) for x in raw_tuple):
        bad_types = {type(x) for x in raw_tuple if not isinstance(x, MulticastPathPlan)}
        raise TypeError(
            f"Unsupported routing choice structure. Expected iterable of MulticastPathPlan, "
            f"got invalid element types: {bad_types}"
        )
    return raw_tuple
