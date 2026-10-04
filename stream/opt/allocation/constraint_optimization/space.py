"""The choices an allocation model decides between, read off the steady-state problem before any variable exists."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from math import ceil, prod
from typing import TYPE_CHECKING, Any, TypeAlias

from stream.cost_model.bandwidth import BandwidthModel, contiguous_span_bytes
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.diagnosis import structural_infeasibility
from stream.opt.allocation.constraint_optimization.utils import get_transfer_latency_for_path
from stream.workload.iterator_type import is_state_operand
from stream.workload.node import HasOutputs, TransferType
from stream.workload.steady_state.iteration_space import IterationVariableType, Reuse
from stream.workload.workload import ComputationNode, HasIterationSpace, InEdge, OutEdge, Tensor, TransferNode

if TYPE_CHECKING:
    from stream.allocation.problem import SteadyStateProblem
    from stream.workload.steady_state.node import Node

Placement: TypeAlias = tuple[Core, ...]
Choice: TypeAlias = tuple[TransferNode, MulticastPathPlan]


class DecisionSpace:
    """What the allocation decides between: the placements each tensor may take, the routes each transfer may
    take and the reuse stops of each tensor, with the facts every family reads off them. Read-only once built."""

    def __init__(self, problem: SteadyStateProblem) -> None:
        self.problem = problem
        self.workload = problem.workload
        self.slot_of = problem.timeslots
        self.accelerator = problem.accelerator
        self.context = problem.transfer_context
        self.offchip_core_id = self.context.offchip_core_id
        self.shared_bandwidth: dict[int, BandwidthModel] = dict(self.accelerator.bandwidth)
        self.iterations = problem.iterations
        self.ssis = problem.ssis
        self.mapping = problem.mapping
        self.cost_lut = problem.cost_lut
        self.max_slot = max(self.slot_of.values()) if self.slot_of else 0
        self.big_m = len(self.workload.nodes()) + 5
        self.force_double_buffering = self.context.force_double_buffering
        self.mem_cores = list(self.context.mem_cores)
        self.ssc_nodes: tuple[ComputationNode, ...] = tuple(self.workload.get_computation_nodes())
        self.transfer_nodes: tuple[TransferNode, ...] = tuple(self.workload.get_transfer_nodes())

        self.tensor_fixed: list[Tensor] = []
        self.tensor_var: list[Tensor] = []
        self.tensor_choices: dict[Tensor, tuple[Placement, ...]] = {}
        self.path_choices: dict[TransferNode, tuple[MulticastPathPlan, ...]] = {}
        self._init_option_sets()
        self._fixed = frozenset(self.tensor_fixed)
        self._candidates: dict[Tensor, set[Core]] = {}
        self._broadcast: dict[TransferNode, bool] = {}
        self._one_memory: dict[TransferNode, bool] = {}

        self.reuse_levels: dict[tuple[Tensor, int], int] = {}
        self.tiles_needed_levels: dict[tuple[Tensor, int], int] = {}
        self.rotation_levels: dict[tuple[Tensor, int], bool] = {}
        self.bds_needed_levels: dict[tuple[Tensor, int], int] = {}
        self.tensors_to_optimize_reuse_for: list[Tensor] = []
        self._ensure_same_ssis_for_all_transfers()
        self._init_transfer_fire_helpers()

        self.link_set: set[CommunicationLink] = set()
        self.links_in_choice: dict[Choice, set[CommunicationLink]] = {}
        self.choice_src_cores: dict[Choice, set[Core]] = {}
        self.choice_dst_cores: dict[Choice, set[Core]] = {}
        self.choice_has_empty_path: dict[Choice, bool] = {}
        self._index_choice_metadata()

    # ------------------------------------------------------------ #
    # option canonicalization                                      #
    # ------------------------------------------------------------ #
    def _init_option_sets(self) -> None:
        for node in self.workload.topological_sort():
            if not isinstance(node, HasOutputs):
                continue
            # A node's outputs, and the state it keeps: the state is resident on the cores the
            # node runs on, so it is allocated exactly where its node is and never moved.
            carried = (
                [x for x in node.inputs if is_state_operand(node, x)] if isinstance(node, HasIterationSpace) else []
            )
            for tensor in (*node.outputs, *carried):
                if tensor in self.tensor_choices:
                    continue
                try:
                    normalized = _normalize_tensor_choices(self.core_allocation(node))
                except ValueError as exc:
                    # The (auto-generated) mapping left this node with no core it can run on -> the mapping
                    # is structurally infeasible. Surface a clean, inspectable diagnosis instead of a bare
                    # error (e.g. auto-mapping onto AIE tiles, which need a hand-written kernel mapping).
                    raise InfeasibleAllocationError(
                        structural_infeasibility(
                            f"node '{getattr(node, 'name', node)}' has no core it can be placed on"
                        )
                    ) from exc
                self.tensor_choices[tensor] = normalized
                if len(normalized) == 1:
                    self.tensor_fixed.append(tensor)
                else:
                    self.tensor_var.append(tensor)

        for tr in self.transfer_nodes:
            self.path_choices[tr] = _normalize_path_choices(self.mapping.get(tr).resource_allocation)

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
                self.rotation_levels[(t, i)] = any(relevancies[i + 1 :])
                self.bds_needed_levels[(t, i)] = 4 if i == len(sizes) - 1 else tiles_factor
            # A second buffer helps only with >1 tile; a loop-invariant tensor (tiles_factor==1) wastes half the memory.
            if self.force_double_buffering and tiles_factor > 1:
                self.tiles_needed_levels[(t, -1)] = 2

    def _index_choice_metadata(self) -> None:
        for tr in self.transfer_nodes:
            for choice in self.path_choices[tr]:
                key = (tr, choice)
                # Data handed over within one memory crosses no link.
                self.links_in_choice[key] = set() if self.in_one_memory(choice) else set(choice.links_used)
                self.link_set.update(self.links_in_choice[key])
                self.choice_src_cores[key] = set(choice.sources)
                self.choice_dst_cores[key] = set(choice.targets)
                self.choice_has_empty_path[key] = len(choice.links_used) == 0

    # ------------------------------------------------------------ #
    # tensors                                                      #
    # ------------------------------------------------------------ #
    def is_fixed(self, t: Tensor) -> bool:
        return t in self._fixed

    def fixed_choice(self, t: Tensor) -> Placement:
        choices = self.tensor_choices[t]
        assert len(choices) == 1, f"Tensor {t.name} is not fixed."
        return choices[0]

    def candidate_cores(self, t: Tensor) -> set[Core]:
        """Every core one of ``t``'s placements uses; the set is shared, so callers do not change it."""
        if (cores := self._candidates.get(t)) is None:
            cores = self._candidates[t] = {core for choice in self.tensor_choices[t] for core in choice}
        return cores

    def stops(self, t: Tensor) -> range:
        """The reuse stops ``t`` may take: -1 (none) up to its outermost applicable temporal loop."""
        return range(-1, len(self.ssis[t].get_applicable_temporal_variables()))

    def core_allocation(self, node: Node) -> tuple[tuple[Core, ...], ...]:
        if isinstance(node, InEdge | OutEdge):
            assert self.offchip_core_id is not None
            return ((self.accelerator.get_core(self.offchip_core_id),),)
        if isinstance(node, TransferNode):
            return self.mapping.get(node).memory_allocation
        return self.mapping.get(node).resource_allocation

    @cached_property
    def sole_outputs(self) -> set[Tensor]:
        """The tensors a transfer moves to a single reader."""
        return {tr.outputs[0] for tr in self.transfer_nodes if len(tr.outputs) == 1}

    def may_single_buffer(self, t: Tensor, stop: int) -> bool:
        """Whether a tensor a transfer moves to one reader may hold its one-tile window at ``stop`` in
        a single buffer where an outer loop moves it on: the next window then waits for the last
        read of this one."""
        return (
            stop >= 0
            and t in self.tensors_to_optimize_reuse_for
            and t in self.sole_outputs
            and self.rotation_levels[(t, stop)]
            and self.tiles_needed_levels[(t, stop)] == 1
        )

    def resident_tiles(self, t: Tensor, stop: int, single: bool = False) -> int:
        """Tiles of this tensor codegen keeps resident when it stops reuse at ``stop``, held in one
        buffer where ``single``."""
        tiles = self.tiles_needed_levels[(t, stop)]
        return max(tiles, 2) if self.rotation_levels.get((t, stop)) and not single else tiles

    def memory_capacity_bits(self, memory: Core) -> int:
        """Bits of ``memory`` left for tensors: its capacity less what the toolchain claims on each core using it."""
        users = [c for c in self.accelerator.core_list if self.accelerator.memory_of(c) == memory]
        return memory.get_memory_capacity() - sum(self.context.reserved_memory_bits(c) for c in users)

    # ------------------------------------------------------------ #
    # transfers                                                    #
    # ------------------------------------------------------------ #
    def is_const_i(self, tr: TransferNode) -> bool:
        return isinstance(next(iter(self.workload.predecessors(tr))), InEdge)

    def is_const_o(self, tr: TransferNode) -> bool:
        return isinstance(next(iter(self.workload.successors(tr))), OutEdge)

    def is_const_io(self, tr: TransferNode) -> bool:
        return self.is_const_i(tr) or self.is_const_o(tr)

    def constant_transfer_tensor(self, tr: TransferNode) -> Tensor:
        if self.is_const_i(tr):
            return tr.outputs[0]
        if self.is_const_o(tr):
            return tr.inputs[0]
        raise ValueError(f"Transfer {tr.name} is not a constant I/O transfer.")

    def transfer_latency_for_path(self, tr: TransferNode, path: MulticastPathPlan) -> int:
        """Cycles one firing of ``tr`` takes on ``path``, before its active fraction and reuse."""
        # A transfer served out of memory the two cores share reads in place: no bytes cross a link,
        # so it adds no time to the slot, the same reason it spends no DMA channel.
        if self.choice_shares_memory(tr, path):
            return 0
        link = get_transfer_latency_for_path(tr, path)
        shared = (self.shared_cycles(core, tr, path, model.contiguous) for core, model in self.shared_bandwidth.items())
        return max(link, *shared) if self.shared_bandwidth else link

    @staticmethod
    def direction(core_id: int, path: MulticastPathPlan) -> str | None:
        """'read' for a transfer out of this core, 'write' for one into it, None for one it takes no part in."""
        if any(c.id == core_id for c in path.sources):
            return "read"
        if any(c.id == core_id for c in path.targets):
            return "write"
        return None

    def shared_cycles(self, core_id: int, tr: TransferNode, path: MulticastPathPlan, rate: float) -> int:
        """Cycles one firing holds a shared-bandwidth core at ``rate``, slowed by its access pattern."""
        direction = self.direction(core_id, path)
        if direction is None:
            return 0
        tensor = tr.inputs[0]
        full = tuple(tensor.subview.source.type.get_shape())
        span = contiguous_span_bytes(tuple(tensor.shape), full, tensor.operand_type.bitwidth)
        return ceil(tensor.size_bits() / (rate * self.shared_bandwidth[core_id].efficiency(span, direction)))

    def offchip_bandwidth(self) -> float:
        """Bits per cycle the array can move across the off-chip boundary."""
        off = self.offchip_core_id
        if off is None:
            return 0.0
        return float(sum(link.bandwidth for link in self.link_set if core_id(link.receiver) == off))

    def placement_width(self, tensors: Iterable[Tensor]) -> int:
        """How many cores one side of a transfer occupies."""
        return max((len(choice) for t in tensors for choice in self.tensor_choices[t]), default=1)

    def distinct_slice_width(self, tensors: Iterable[Tensor]) -> int:
        """How many distinct slices one side of a transfer holds.

        Cores that a spatial loop does not address separately read the same slice, and the
        object-fifo lowering serves them from one channel, so they do not widen the fan-out.
        """
        return max(
            (
                prod(v.size for v in self.ssis[t].variables if v.type is IterationVariableType.SPATIAL and v.relevant)
                for t in tensors
                if t in self.ssis
            ),
            default=1,
        )

    def transfer_is_broadcast(self, tr: TransferNode) -> bool:
        """Whether several cores are served the same slice, so one fifo carries them all.

        ``requiresDMAs`` bails out before it ever looks at the tiles unless the fifo has a
        single consumer, so a broadcast is on the DMA however the cores are placed.
        """
        if (broadcast := self._broadcast.get(tr)) is None:
            tensors = unique_tensors(tr.outputs)
            broadcast = self._broadcast[tr] = self.distinct_slice_width(tensors) < self.placement_width(tensors)
        return broadcast

    def transfer_shares_memory(self, tr: TransferNode, core: Core, incoming: bool) -> bool:
        """Whether this core is served this transfer out of memory it already shares.

        The object-fifo lowering keeps a fifo out of the DMA when it has one consumer, no
        repeat count and no layout transform on the way. Stream only routes a transfer core
        to core when the two sides already agree on layout -- a disagreement is what puts it
        on a memory tile -- so the case left to check is whether the cores this one actually
        hands to, or takes from, are its neighbours.
        """
        if self.within_one_memory(tr):
            return True
        choices = self.path_choices.get(tr) or ()
        if not choices or self.transfer_is_broadcast(tr):
            return False
        # Only a transfer that lands straight on the cores is lowered core to core. One
        # staged on a memory tile is two transfers, and each leg ends on the tile.
        if tr.transfer_type is not TransferType.COMPUTE_TO_COMPUTE:
            return False
        for choice in choices:
            touching = [(a, b) for a, b in communicating_pairs(choice) if (b if incoming else a) == core]
            if not touching:
                return False
            if any(not self.context.shares_memory(one, other) for one, other in touching):
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
        pairs = communicating_pairs(choice)
        return bool(pairs) and all(self.context.shares_memory(one, other) for one, other in pairs)

    def in_one_memory(self, choice: MulticastPathPlan) -> bool:
        """Whether every core of this choice uses one memory, so the data it hands over never moves."""
        return len({self.accelerator.memory_of(c) for c in (*choice.sources, *choice.targets)}) == 1

    def within_one_memory(self, tr: TransferNode) -> bool:
        """Whether every placement of this transfer stays in one memory."""
        if (within := self._one_memory.get(tr)) is None:
            choices = self.path_choices.get(tr)
            within = self._one_memory[tr] = bool(choices) and all(self.in_one_memory(choice) for choice in choices)
        return within

    # ------------------------------------------------------------ #
    # kernel state handed between cores                            #
    # ------------------------------------------------------------ #
    @cached_property
    def handovers(self) -> tuple[tuple[Core, Core, int], ...]:
        """Cores that pass a kernel's state to the step behind them, and the bits each holds.

        A kernel that keeps a running reduction hands the finished scale to whichever core
        consumes its output, in a buffer both ends hold. Which core meets which is the same
        relation the transfer between them uses, so it is read off the allocation the mapping
        already declares rather than being decided again.
        """
        found: list[tuple[Core, Core, int]] = []
        for node in self.ssc_nodes:
            kernel = self.mapping.get(node).kernel
            for state in kernel.state_operands() if kernel else ():
                if not state.handover:
                    continue
                held = next((t for t in node.inputs if is_state_operand(node, t)), None)
                if held is None:
                    continue
                bits = self.workload.get_tensor_single_core(held, node, self.mapping).size_bits()
                sources = self.core_allocation(node)[0]
                for consumer in self._consumers(node):
                    targets = self.core_allocation(consumer)[0]
                    narrow = min(len(sources), len(targets))
                    found += [
                        (sources[i], targets[j], state.handover * bits)
                        for i in range(len(sources))
                        for j in range(len(targets))
                        if narrow and i % narrow == j % narrow
                    ]
        return tuple(found)

    def _consumers(self, node: ComputationNode) -> list[ComputationNode]:
        """The computation nodes this one's output reaches, across the transfer between them."""
        reached = []
        for succ in self.workload.successors(node):
            reached += (
                [succ]
                if isinstance(succ, ComputationNode)
                else [x for x in self.workload.successors(succ) if isinstance(x, ComputationNode)]
            )
        return reached


def core_id(end: Core | str) -> int | None:
    """A link end's core id, None for an end that is not a core."""
    return end.id if isinstance(end, Core) else None


def unique_tensors(tensors: Iterable[Any]) -> list[Tensor]:
    """The tensors among ``tensors``, each once, in order."""
    return list(dict.fromkeys(t for t in tensors if isinstance(t, Tensor)))


def communicating_pairs(choice: MulticastPathPlan) -> tuple[tuple[Core, Core], ...]:
    """Which source and target of this transfer actually hand to one another.

    Codegen matches a producer to a consumer by spatial index, and the spatial part of
    a split runs fastest, so the consumer holding spatial point ``j`` is fed by the
    producers at ``j``, ``j + m``, ``j + 2m`` and so on, ``m`` being the narrower of
    the two sides. Which is the same relation the flash bindings use to find the core
    holding the other half of an online-softmax step.
    """
    src, dst = choice.sources, choice.targets
    if not src or not dst:
        return ()
    narrow = min(len(src), len(dst))
    return tuple((src[i], dst[j]) for i in range(len(src)) for j in range(len(dst)) if i % narrow == j % narrow)


def _normalize_tensor_choices(raw: Any) -> tuple[Placement, ...]:
    """The placements a node's allocation offers, as a tuple of core tuples."""
    if raw is None:
        raise ValueError("Tensor allocation options cannot be None.")
    raw_tuple = tuple(raw)
    if not raw_tuple:
        raise ValueError("Tensor allocation options cannot be empty.")
    if all(isinstance(x, Core) for x in raw_tuple):
        return (tuple(raw_tuple),)
    out: list[Placement] = []
    for choice in raw_tuple:
        choice_tuple = tuple(choice)
        if not choice_tuple:
            raise ValueError("Empty tensor placement choice encountered.")
        if not all(isinstance(c, Core) for c in choice_tuple):
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
