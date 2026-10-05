"""How long each slot lasts: the slowest node or transfer in it."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE
from stream.opt.allocation.constraint_optimization.utils import get_active_latency

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.opt.solver import SolverVar
    from stream.workload.workload import TransferNode


def reuse_selectors(ctx: FormulationContext, t: object) -> list[tuple[SolverVar, float]]:
    """Each reuse stop of ``t`` with the reuse factor it gives, for backends that cannot divide by a variable."""
    z_stop, levels = ctx.vars.z_stop, ctx.space.reuse_levels
    return [(z_stop[(t, s)], float(levels[(t, s)])) for s in ctx.space.stops(t)]


class SlotLatency:
    """A slot lasts as long as the slowest node or transfer in it; the longest any can take bounds every slot."""

    name: ClassVar[str] = "slot_latency"
    requires: ClassVar[tuple[str, ...]] = ("reuse_factor",)
    provides: ClassVar[tuple[str, ...]] = ("transfer_latency", SLOT_PRESSURE)

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, model, slot_latency = ctx.space, ctx.model, ctx.vars.slot_latency
        for n in space.ssc_nodes:
            model.add_constr(slot_latency[space.slot_of[n]] >= space.active_runtime(n), name=f"ssc_lat_{n.name}")
        for (tr, choice), y in ctx.vars.y.items():
            latency = _active_transfer_latency(ctx, tr, choice, y)
            q.add("transfer_latency", latency._raw, index=(tr, choice))
            model.add_constr(slot_latency[space.slot_of[tr]] >= latency, name=f"tr_lat_{tr.name}_{hash(choice)}")
        q.add(SLOT_PRESSURE, 0, index="slot_latency", upper_bound=_longest_step(space))


def _active_transfer_latency(
    ctx: FormulationContext, tr: TransferNode, choice: MulticastPathPlan, y: SolverVar
) -> SolverVar:
    """Cycles ``tr`` takes per iteration on ``choice`` while ``y`` chooses it: its active latency over its reuse."""
    q = ctx.quantities
    latency_constant = float(ctx.space.transfer_latency_for_path(tr, choice))
    return ctx.binary_times_const_over_linexpr(
        binary_var=y,
        numerator=get_active_latency(tr, latency_constant, ctx.space.ssis),
        denominator_expr=q.get("reuse_factor", tr).expr,
        denominator_lb=1.0,
        base_name=f"transfer_latency_{tr}",
        selectors=reuse_selectors(ctx, tr.outputs[0]),
    )


def _longest_step(space: DecisionSpace) -> int:
    """The most cycles any node or transfer takes in a slot, on any core or path."""
    longest = 0
    for n in space.ssc_nodes:
        longest = max(longest, space.runtime(n))
    for tr in space.transfer_nodes:
        for choice in space.path_choices[tr]:
            longest = max(longest, ceil(space.transfer_latency_for_path(tr, choice)))
    return longest
