"""The DMA channels each core's transfers drive in and out."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind
from stream.opt.allocation.constraint_optimization.families import LATENCY
from stream.opt.allocation.constraint_optimization.space import unique_tensors
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import ObjectiveLevel, SolverVar, SolverVarType
from stream.workload.workload import Tensor

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.workload import TransferNode


DMA_CHANNELS = ResourceKind("dma_channels", "DMA channel limit exceeded")


class DmaChannels:
    """The DMA channels each core's transfers drive in and out, whose peaks the latency objective charges; a
    namespace family bounds them. Incoming channels follow the transfer outputs a core holds, outgoing ones the
    inputs; a fifo carries every slice its core works through, so a transfer costs a core its fan, not its slices."""

    name: ClassVar[str] = "dma_channels"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("dma_in", "dma_out", "dma_peak_in", "dma_peak_out")

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, model = ctx.space, ctx.model
        handover_out: dict[Core, int] = defaultdict(int)
        handover_in: dict[Core, int] = defaultdict(int)
        for one, other, _ in space.handovers:
            if not space.hardware.shares_memory(one, other):
                handover_out[one] += 1
                handover_in[other] += 1

        dma_cores = _dma_candidate_cores(space)
        core_dma_in: dict[Core, SolverVar] = {}
        core_dma_out: dict[Core, SolverVar] = {}
        for core in dma_cores:
            v_in = model.add_var(vtype=SolverVarType.INTEGER, name=f"coreDmaIn_{resource_key(core)}")
            v_out = model.add_var(vtype=SolverVarType.INTEGER, name=f"coreDmaOut_{resource_key(core)}")
            in_expr = model.quicksum(_channels(ctx, tr, core, True) for tr in space.transfer_nodes)
            out_expr = model.quicksum(_channels(ctx, tr, core, False) for tr in space.transfer_nodes)
            in_expr = in_expr + handover_in[core]
            out_expr = out_expr + handover_out[core]
            ctx.add_constr(v_in == in_expr, name=f"coreDmaInConstr_{resource_key(core)}", resource=core)
            ctx.add_constr(v_out == out_expr, name=f"coreDmaOutConstr_{resource_key(core)}", resource=core)
            core_dma_in[core] = v_in
            core_dma_out[core] = v_out

        max_in = model.add_var(vtype=SolverVarType.INTEGER, name="maxCoreDmaIn")
        max_out = model.add_var(vtype=SolverVarType.INTEGER, name="maxCoreDmaOut")
        for core in dma_cores:
            ctx.add_constr(max_in >= core_dma_in[core], name=f"maxCoreDmaIn_lb_{resource_key(core)}", resource=core)
            ctx.add_constr(max_out >= core_dma_out[core], name=f"maxCoreDmaOut_lb_{resource_key(core)}", resource=core)

        for core, usage in core_dma_in.items():
            q.add("dma_in", usage, index=core)
        for core, usage in core_dma_out.items():
            q.add("dma_out", usage, index=core)
        q.add("dma_peak_in", max_in._raw)
        q.add("dma_peak_out", max_out._raw)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q = ctx.quantities
        peaks = q.get("dma_peak_in").expr + q.get("dma_peak_out").expr
        return [ObjectiveLevel(expr=peaks, priority=LATENCY, name="latency")]


def _dma_candidate_cores(space: DecisionSpace) -> set[Core]:
    """The on-chip cores that may hold a tensor a transfer moves; the off-chip core spends no channel."""
    return {
        core
        for t in space.workload.tensors
        if isinstance(t, Tensor)
        for core in space.candidate_cores(t)
        if core.id != space.offchip_core_id
    }


def _channels(ctx: FormulationContext, tr: TransferNode, core: Core, incoming: bool) -> Any:
    """The channels one transfer drives on one core: its fan wherever the core holds a tensor it moves, none
    where the core is served out of memory it shares."""
    space = ctx.space
    if space.transfer_shares_memory(tr, core, incoming=incoming):
        return 0
    return _fan(space, tr, incoming) * _side_uses_core_var(ctx, tr, core, incoming)._raw


def _fan(space: DecisionSpace, tr: TransferNode, incoming: bool) -> int:
    """DMA channels a destination core is fed by (a fifo per source gathering into it), or a source core
    drives (a fifo per destination it feeds a distinct slice to), or per core whose tile overlaps a window."""
    if overlaps := space.overlaps(tr):
        return max(Counter(j if incoming else i for i, j in overlaps).values())
    if incoming:
        return max(
            1, space.placement_width(unique_tensors(tr.inputs)) // space.placement_width(unique_tensors(tr.outputs))
        )
    n_src = space.distinct_slice_width(unique_tensors(tr.inputs))
    return max(1, space.distinct_slice_width(unique_tensors(tr.outputs)) // n_src)


def _side_uses_core_var(ctx: FormulationContext, tr: TransferNode, core: Core, incoming: bool) -> SolverVar:
    """Whether any tensor this transfer brings to (or takes from) this core sits on it."""
    model = ctx.model
    side = "in" if incoming else "out"
    name = f"{tr.name}_{resource_key(core)}_{side}"
    u = model.add_var(vtype=SolverVarType.BINARY, name=f"us_{name}")
    tensors = unique_tensors(tr.outputs if incoming else tr.inputs)
    occ_exprs = [ctx.tensor_on_core_expr(t, core) for t in tensors]
    if not occ_exprs:
        ctx.add_constr(u == 0, name=f"us_zero_{name}", resource=core)
        return u
    for i, occ in enumerate(occ_exprs):
        ctx.add_constr(u >= occ, name=f"us_lb_{name}_{i}", resource=core)
    ctx.add_constr(u <= model.quicksum(occ_exprs), name=f"us_ub_{name}", resource=core)
    return u
