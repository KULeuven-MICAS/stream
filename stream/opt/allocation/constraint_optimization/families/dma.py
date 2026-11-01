"""The DMA channels each memory's transfers drive in and out."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import READ, WRITE, Port
from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind
from stream.opt.allocation.constraint_optimization.families import DMA_PEAKS
from stream.opt.allocation.constraint_optimization.space import unique_tensors
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import ObjectiveLevel, SolverVar, SolverVarType
from stream.workload.workload import Tensor

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.workload import TransferNode


DMA_CHANNELS = ResourceKind("dma_channels", "DMA channel limit exceeded")


class DmaChannels:
    """The DMA channels each memory's transfers drive in and out. A namespace family bounds them, and an objective
    level after the off-chip traffic keeps the peak over the on-chip memories low. Incoming channels follow the
    transfer outputs a memory holds, outgoing ones the inputs; one fifo carries every slice, so a memory pays its fan,
    not its slices. The off-chip memory's channels are counted per DMA port and take no part in the peak."""

    name: ClassVar[str] = "dma_channels"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = (
        "dma_in",
        "dma_out",
        "dma_peak_in",
        "dma_peak_out",
        "dma_port_in",
        "dma_port_out",
    )

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, model = ctx.space, ctx.model
        handover_out: dict[Memory, int] = defaultdict(int)
        handover_in: dict[Memory, int] = defaultdict(int)
        for one, other, _ in space.handovers:
            if not space.hardware.shares_memory(one, other):
                handover_out[one] += 1
                handover_in[other] += 1

        memory_dma_in: dict[Memory, SolverVar] = {}
        memory_dma_out: dict[Memory, SolverVar] = {}
        for memory in _dma_candidate_memories(space):
            v_in = model.add_var(vtype=SolverVarType.INTEGER, name=f"memoryDmaIn_{resource_key(memory)}")
            v_out = model.add_var(vtype=SolverVarType.INTEGER, name=f"memoryDmaOut_{resource_key(memory)}")
            in_expr = model.quicksum(_channels(ctx, tr, memory, True) for tr in space.transfer_nodes)
            out_expr = model.quicksum(_channels(ctx, tr, memory, False) for tr in space.transfer_nodes)
            in_expr = in_expr + handover_in[memory]
            out_expr = out_expr + handover_out[memory]
            ctx.add_constr(v_in == in_expr, name=f"memoryDmaInConstr_{resource_key(memory)}", resource=memory)
            ctx.add_constr(v_out == out_expr, name=f"memoryDmaOutConstr_{resource_key(memory)}", resource=memory)
            memory_dma_in[memory] = v_in
            memory_dma_out[memory] = v_out

        max_in = model.add_var(vtype=SolverVarType.INTEGER, name="maxMemoryDmaIn")
        max_out = model.add_var(vtype=SolverVarType.INTEGER, name="maxMemoryDmaOut")
        for memory, v_in in memory_dma_in.items():
            key = resource_key(memory)
            ctx.add_constr(max_in >= v_in, name=f"maxMemoryDmaIn_lb_{key}", resource=memory)
            ctx.add_constr(max_out >= memory_dma_out[memory], name=f"maxMemoryDmaOut_lb_{key}", resource=memory)

        for memory, usage in memory_dma_in.items():
            q.add("dma_in", usage, index=memory)
        for memory, usage in memory_dma_out.items():
            q.add("dma_out", usage, index=memory)
        q.add("dma_peak_in", max_in._raw)
        q.add("dma_peak_out", max_out._raw)
        for direction, usage_per_port in zip(("in", "out"), _port_channels(ctx), strict=True):
            for (memory, port), usage in usage_per_port.items():
                q.add(f"dma_port_{direction}", usage, index=(memory, port))

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q = ctx.quantities
        peaks = q.get("dma_peak_in").expr + q.get("dma_peak_out").expr
        return [ObjectiveLevel(expr=peaks, priority=DMA_PEAKS, name="dma_peaks")]


def _port_channels(ctx: FormulationContext) -> tuple[dict[tuple[Memory, Port], Any], ...]:
    """The channels transfers drive into and out of each DMA port of each off-chip memory, keyed by (memory, port).
    Each link of a route passes one DMA port (:meth:`Memory.dma_port`); a route spreads the transfer's fan over its
    ports in proportion to the bits each carries, with at least one fifo per port. Channels are held for the whole
    run, so a memory of an edge's placement that the iterations walk the split axis through holds its fifos even where
    the busiest iteration does not read it."""
    space = ctx.space
    usage: tuple[dict[tuple[Memory, Port], Any], ...] = (defaultdict(int), defaultdict(int))
    for memory in space.accelerator.offchip_memories:
        for tr in space.transfer_nodes:
            for choice in space.path_choices[tr]:
                link_shares = dict(choice.link_shares)
                walked = space.walk_shares(tr, choice).get(memory, 0.0) > 0
                for incoming, side in ((True, choice.targets), (False, choice.sources)):
                    if memory not in side:
                        continue
                    direction = WRITE if incoming else READ
                    operand = space.operand_role(tr, read_side=not incoming)
                    carried: dict[Port, float] = defaultdict(float)
                    links = choice.links_at(memory, leaving=not incoming)
                    for link in links or (space.memory_links(memory, not incoming) if walked else ()):
                        dma = memory.dma_port(link.ports[memory], direction, operand)
                        if dma is not None:
                            carried[dma] += link_shares.get(link, 0.0)
                    total, fan = sum(carried.values()), _memory_fan(space, tr, choice, memory, incoming)
                    for port, bits in carried.items():
                        fifos = max(1, ceil(fan * (bits / total if total else 1.0 / len(carried))))
                        usage[0 if incoming else 1][(memory, port)] += fifos * ctx.vars.y[(tr, choice)]._raw
    return usage


def _dma_candidate_memories(space: DecisionSpace) -> set[Memory]:
    """The on-chip memories that may hold a tensor a transfer moves; the off-chip memory's channels are counted per
    DMA port in :func:`_port_channels`."""
    return {
        memory
        for t in space.workload.tensors
        if isinstance(t, Tensor)
        for memory in space.candidate_memories(t)
        if not memory.offchip
    }


def _channels(ctx: FormulationContext, tr: TransferNode, memory: Memory, incoming: bool) -> Any:
    """The channels one transfer drives on one memory: its fan where the memory holds a tensor it moves, none where
    the data does not need to move."""
    space = ctx.space
    if space.transfer_shares_memory(tr, memory, incoming=incoming):
        return 0
    return _fan(space, tr, incoming) * _side_uses_memory_var(ctx, tr, memory, incoming)._raw


def _fan(space: DecisionSpace, tr: TransferNode, incoming: bool) -> int:
    """DMA channels a destination memory is fed by (a fifo per source gathering into it), or a source memory
    drives (a fifo per destination it feeds a distinct slice to). With overlapping windows, one per memory it shares
    no memory with whose tile overlaps a window."""
    if space.overlaps(tr):
        counts = [
            Counter(b if incoming else a for a, b in space.pairs(tr, choice) if not space.hardware.shares_memory(a, b))
            for choice in space.path_choices[tr]
        ]
        return max((n for count in counts for n in count.values()), default=0)
    if incoming:
        return max(
            1, space.placement_width(unique_tensors(tr.inputs)) // space.placement_width(unique_tensors(tr.outputs))
        )
    n_src = space.distinct_slice_width(unique_tensors(tr.inputs))
    return max(1, space.distinct_slice_width(unique_tensors(tr.outputs)) // n_src)


def _memory_fan(
    space: DecisionSpace, tr: TransferNode, choice: MulticastPathPlan, memory: Memory, incoming: bool
) -> int:
    """:func:`_fan`, or, where other memories share ``memory``'s side of ``choice``, at most the positions it exchanges
    data with."""
    fan = _fan(space, tr, incoming)
    if set(choice.targets if incoming else choice.sources) == {memory}:
        return fan
    side, own = (choice.targets, 1) if incoming else (choice.sources, 0)
    partners = {pair[1 - own] for pair in choice.pairs if side[pair[own]] is memory}
    return min(fan, max(1, len(partners)))


def _side_uses_memory_var(ctx: FormulationContext, tr: TransferNode, memory: Memory, incoming: bool) -> SolverVar:
    """Whether any tensor this transfer brings to (or takes from) this memory sits in it."""
    model = ctx.model
    side = "in" if incoming else "out"
    name = f"{tr.name}_{resource_key(memory)}_{side}"
    u = model.add_var(vtype=SolverVarType.BINARY, name=f"us_{name}")
    tensors = unique_tensors(tr.outputs if incoming else tr.inputs)
    occ_exprs = [ctx.tensor_in_memory_expr(t, memory) for t in tensors]
    if not occ_exprs:
        ctx.add_constr(u == 0, name=f"us_zero_{name}", resource=memory)
        return u
    for i, occ in enumerate(occ_exprs):
        ctx.add_constr(u >= occ, name=f"us_lb_{name}_{i}", resource=memory)
    ctx.add_constr(u <= model.quicksum(occ_exprs), name=f"us_ub_{name}", resource=memory)
    return u
