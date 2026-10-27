"""The DMA channels each memory's transfers drive in and out."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.ports import READ, WRITE
from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind
from stream.opt.allocation.constraint_optimization.families import DMA_PEAKS
from stream.opt.allocation.constraint_optimization.space import unique_tensors
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import ObjectiveLevel, SolverVar, SolverVarType
from stream.workload.workload import Tensor

if TYPE_CHECKING:
    from stream.hardware.architecture.memory import Memory
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.workload import TransferNode


DMA_CHANNELS = ResourceKind("dma_channels", "DMA channel limit exceeded")


class DmaChannels:
    """The DMA channels each memory's transfers drive in and out, whose peaks over the memories on chip an objective
    level after the off-chip traffic keeps low; a namespace family bounds them. Incoming channels follow the transfer
    outputs a memory holds, outgoing ones the inputs; a fifo carries every slice its memory serves, so a memory pays its
    fan, not slices. The off-chip memory spends channels only where its ports are a DMA's, as an AIE2 shim's are, and
    then counts in no peak: every transfer in or out of the chip shares it."""

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

        dma_memories = _dma_candidate_memories(space)
        memory_dma_in: dict[Memory, SolverVar] = {}
        memory_dma_out: dict[Memory, SolverVar] = {}
        for memory in (memory for memory in dma_memories if not memory.offchip):  # the off-chip memory's are per port
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
        for direction, usage_per_port in zip(("in", "out"), _port_channels(ctx, dma_memories), strict=True):
            for (memory, port), usage in usage_per_port.items():
                q.add(f"dma_port_{direction}", usage, index=(memory, port))

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q = ctx.quantities
        peaks = q.get("dma_peak_in").expr + q.get("dma_peak_out").expr
        return [ObjectiveLevel(expr=peaks, priority=DMA_PEAKS, name="dma_peaks")]


def _port_channels(ctx: FormulationContext, memories: set[Memory]) -> tuple[dict[tuple[Memory, str], Any], ...]:
    """Per DMA port of the off-chip memory (a port declaring channels, one shim's), by name, the channels the
    transfers drive into it and out of it: each route a transfer may take enters or leaves the memory over its links,
    one DMA port per link (:meth:`Memory.dma_port`), so a route's choice spends the transfer's fan over those ports as
    the route's bits split over its links, a fifo at least on each. The DMA's channels are its own, whether or not the
    memory's bandwidth is measured. A tile's memory has one DMA, so its channels are counted per memory."""
    space = ctx.space
    usage: tuple[dict[tuple[Memory, str], Any], ...] = (defaultdict(int), defaultdict(int))
    for memory in (memory for memory in memories if memory.offchip):
        for tr in space.transfer_nodes:
            for choice in space.path_choices[tr]:
                link_shares = dict(choice.link_shares)
                for incoming, side in ((True, choice.targets), (False, choice.sources)):
                    if memory not in side:
                        continue
                    direction = WRITE if incoming else READ
                    operand = space.operand_role(tr, read_side=not incoming)
                    carried: dict[str, float] = defaultdict(float)
                    for link in choice.links_at(memory, leaving=not incoming):
                        if (dma := memory.dma_port(link.ports[memory], direction, operand)) is not None:
                            carried[dma[0]] += link_shares.get(link, 0.0)
                    total, fan = sum(carried.values()), _fan(space, tr, incoming)
                    for name, bits in carried.items():
                        fifos = max(1, ceil(fan * (bits / total if total else 1.0 / len(carried))))
                        usage[0 if incoming else 1][(memory, name)] += fifos * ctx.vars.y[(tr, choice)]._raw
    return usage


def _dma_candidate_memories(space: DecisionSpace) -> set[Memory]:
    """The memories that may hold a tensor a transfer moves and spend a DMA channel doing so: those on chip, and the
    off-chip memory where its ports are DMA channels; one without spends none."""
    return {
        memory
        for t in space.workload.tensors
        if isinstance(t, Tensor)
        for memory in space.candidate_memories(t)
        if not memory.offchip or any(memory.dma_channels(direction) is not None for direction in (READ, WRITE))
    }


def _channels(ctx: FormulationContext, tr: TransferNode, memory: Memory, incoming: bool) -> Any:
    """The channels one transfer drives on one memory: its fan wherever the memory holds a tensor it moves, none
    where the memory is served out of memory it shares."""
    space = ctx.space
    if space.transfer_shares_memory(tr, memory, incoming=incoming):
        return 0
    return _fan(space, tr, incoming) * _side_uses_memory_var(ctx, tr, memory, incoming)._raw


def _fan(space: DecisionSpace, tr: TransferNode, incoming: bool) -> int:
    """DMA channels a destination memory is fed by (a fifo per source gathering into it), or a source memory
    drives (a fifo per destination it feeds a distinct slice to), or per memory it shares nothing with whose tile
    overlaps a window."""
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
