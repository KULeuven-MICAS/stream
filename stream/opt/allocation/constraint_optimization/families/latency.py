"""How long each slot lasts: the slowest node or transfer in it."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE
from stream.opt.allocation.constraint_optimization.utils import get_active_latency
from stream.opt.solver import SolverVarType

if TYPE_CHECKING:
    from collections.abc import Callable

    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.noc.communication_link import CommunicationLink
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import CopySet, DecisionSpace
    from stream.opt.solver import LinExpr, SolverVar
    from stream.workload.workload import Tensor, TransferNode


def reuse_selectors(ctx: FormulationContext, t: object) -> list[tuple[SolverVar, float]]:
    """Each reuse stop of ``t`` with the reuse factor it gives, for backends that cannot divide by a variable."""
    z_stop, levels = ctx.vars.z_stop, ctx.space.reuse_levels
    return [(z_stop[(t, s)], float(levels[(t, s)])) for s in ctx.space.stops(t)]


class SlotLatency:
    """A slot lasts as long as the slowest node or transfer in it; the longest any can take bounds every slot."""

    name: ClassVar[str] = "slot_latency"
    requires: ClassVar[tuple[str, ...]] = ("copy_reuse_factor",)
    provides: ClassVar[tuple[str, ...]] = (
        "transfer_latency",
        "copy_latency",
        "copy_firing_rate",
        "shared_firing_rate",
        SLOT_PRESSURE,
    )

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, model, slot_latency = ctx.space, ctx.model, ctx.vars.slot_latency
        for n in space.ssc_nodes:
            model.add_constr(slot_latency[space.slot_of[n]] >= space.active_runtime(n), name=f"ssc_lat_{n.name}")
        for (tr, choice), y in ctx.vars.y.items():
            if not is_single_copy(space.copy_sets(tr, choice)):
                _add_copy_firing_rates(ctx, tr, choice, y)
            terms = per_copy_terms(
                ctx, tr, choice, y, partial(space.transfer_latency_for_path, tr, choice), f"transfer_latency_{tr}"
            )
            if len(terms) > 1:
                for copies, term in terms.items():
                    q.add("copy_latency", term._raw, index=(tr, choice, copies))
            latency = next(iter(terms.values())) if len(terms) == 1 else _busiest_resource(ctx, tr, choice)
            q.add("transfer_latency", latency._raw, index=(tr, choice))
            model.add_constr(slot_latency[space.slot_of[tr]] >= latency, name=f"tr_lat_{space.route_name(tr, choice)}")
        q.add(SLOT_PRESSURE, 0, index="slot_latency", upper_bound=_longest_step(space))


def per_copy_terms(  # noqa: PLR0913
    ctx: FormulationContext,
    tr: TransferNode,
    choice: MulticastPathPlan,
    y: SolverVar,
    cycles: Callable[[CopySet | None], float],
    base_name: str,
) -> dict[CopySet | None, SolverVar | LinExpr]:
    """Cycles per iteration ``tr`` keeps a resource busy on ``choice`` while ``y`` chooses it, from the ``cycles`` one
    firing takes: keyed None over its one copy's reuse, else per copy set at that set's rate."""
    groups = ctx.space.copy_sets(tr, choice)
    if is_single_copy(groups):
        copy = tr.outputs[groups[0][0]] if groups else tr.outputs[0]
        return {None: _over_reuse(ctx, tr, y, cycles(None), copy, base_name)}
    rates = copy_firing_rates(ctx, tr, choice)
    return {copies: ctx.model.quicksum([cycles(copies) * rate]) for copies, rate in rates.items()}


def is_single_copy(groups: tuple[CopySet, ...]) -> bool:
    """Whether one copy reads all a route delivers, so its reuse alone times it."""
    return len(groups) <= 1 and all(len(copies) == 1 for copies in groups)


def copy_firing_rates(ctx: FormulationContext, tr: TransferNode, choice: MulticastPathPlan) -> dict[CopySet, Any]:
    """Per copy set ``choice`` delivers to, its firings per iteration while chosen."""
    q = ctx.quantities
    return {copies: q.get("copy_firing_rate", (tr, choice, copies)).expr for copies in ctx.space.copy_sets(tr, choice)}


def _add_copy_firing_rates(ctx: FormulationContext, tr: TransferNode, choice: MulticastPathPlan, y: SolverVar) -> None:
    """Build :func:`copy_firing_rates` for a route ``y`` chooses."""
    name = f"copy_firing_rate_{ctx.space.route_name(tr, choice)}"
    for k, copies in enumerate(ctx.space.copy_sets(tr, choice)):
        rate = (
            _over_reuse(ctx, tr, y, 1.0, tr.outputs[copies[0]], f"{name}_{k}")
            if len(copies) == 1
            else _over_shared_rate(ctx, tr, y, 1.0, [tr.outputs[c] for c in copies], f"{name}_{k}")
        )
        ctx.quantities.add("copy_firing_rate", rate._raw, index=(tr, choice, copies))


def cycles_per_copy_set(
    space: DecisionSpace, tr: TransferNode, choice: MulticastPathPlan
) -> list[dict[CopySet, float]]:
    """Per link and shared-bandwidth memory of ``choice``, the cycles one firing of each copy set holds it."""
    if space.choice_shares_memory(tr, choice):
        return []
    groups = space.copy_sets(tr, choice)
    per_link: dict[CommunicationLink, dict[CopySet, float]] = defaultdict(dict)
    for copies in groups:
        for link, bits in space.copy_set_link_bits(tr, choice, copies).items():
            per_link[link][copies] = bits / link.bandwidth
    shared = [
        {copies: float(space.shared_cycles(memory, tr, choice, model.contiguous, copies)) for copies in groups}
        for memory, model in space.shared_bandwidth.items()
    ]
    return [*per_link.values(), *shared]


def _busiest_resource(ctx: FormulationContext, tr: TransferNode, choice: MulticastPathPlan) -> SolverVar:
    """Cycles per iteration ``tr`` takes on a route delivering to several copy sets: those of its busiest link
    or shared memory, each set at its own rate."""
    rates = copy_firing_rates(ctx, tr, choice)
    held = cycles_per_copy_set(ctx.space, tr, choice)
    bound = max((sum(cycles.values()) for cycles in held), default=0.0)
    name = ctx.space.route_name(tr, choice)
    latency = ctx.model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=bound, name=f"transfer_latency_{name}")
    for k, cycles in enumerate(held):
        carried = ctx.model.quicksum(c * rates[copies] for copies, c in cycles.items())
        ctx.model.add_constr(latency >= carried, name=ctx.unique_name(f"transfer_latency_{name}_{k}"))
    return latency


def copy_latencies(ctx: FormulationContext) -> dict[tuple[TransferNode, MulticastPathPlan], dict[CopySet | None, Any]]:
    """Per transfer and route, its cycles per iteration per copy set, or keyed None its whole latency."""
    q = ctx.quantities
    per_copies: dict[tuple[TransferNode, MulticastPathPlan], dict[CopySet | None, Any]] = defaultdict(dict)
    for (tr, choice, copies), quantity in (q.indexed("copy_latency") if "copy_latency" in q else {}).items():
        per_copies[(tr, choice)][copies] = quantity.expr
    return {
        key: per_copies.get(key) or {None: quantity.expr} for key, quantity in q.indexed("transfer_latency").items()
    }


def link_busy(ctx: FormulationContext) -> dict[tuple[CommunicationLink, int], list[Any]]:
    """Per link and slot, the cycles per iteration each route in that slot keeps the link busy."""
    space = ctx.space
    busy: dict[tuple[CommunicationLink, int], list[Any]] = defaultdict(list)
    for (tr, choice), terms in copy_latencies(ctx).items():
        for copies, latency in terms.items():
            for link, share in space.link_load(tr, choice, copies).items():
                busy[(link, space.slot_of[tr])].append(share * latency)
    return busy


def _over_reuse(
    ctx: FormulationContext, tr: TransferNode, y: SolverVar, cycles: float, copy: Tensor, base_name: str
) -> SolverVar:
    """``y`` times the active ``cycles`` over ``copy``'s reuse factor."""
    return ctx.binary_times_const_over_linexpr(
        binary_var=y,
        numerator=get_active_latency(tr, cycles, ctx.space.ssis),
        denominator_expr=ctx.quantities.get("copy_reuse_factor", copy).expr,
        denominator_lb=1.0,
        base_name=base_name,
        selectors=reuse_selectors(ctx, copy),
    )


def _over_shared_rate(  # noqa: PLR0913
    ctx: FormulationContext, tr: TransferNode, y: SolverVar, cycles: float, copies: list[Tensor], base_name: str
) -> SolverVar:
    """``y`` times the active ``cycles`` over the smallest reuse factor of ``copies``."""
    numerator = get_active_latency(tr, cycles, ctx.space.ssis)
    scaled = ctx.model.quicksum([numerator * _shared_firing_rate(ctx, copies)._raw])
    return ctx.binary_scaled_continuous(
        binary_var=y, continuous_var=scaled, continuous_ub=numerator, base_name=f"{base_name}_gated"
    )


def _shared_firing_rate(ctx: FormulationContext, copies: list[Tensor]) -> LinExpr:
    """One over the smallest reuse factor of ``copies``: the firings per iteration of a box they all read."""
    q, model, space, z_stop = ctx.quantities, ctx.model, ctx.space, ctx.vars.z_stop
    key = tuple(copies)
    if "shared_firing_rate" in q and key in q.indexed("shared_firing_rate"):
        return q.get("shared_firing_rate", key).expr
    name = ctx.unique_name(f"shared_firing_rate_{copies[0].name}")
    rate = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=1.0, name=name)
    fires = {(copy, stop): 1 / space.reuse_levels[(copy, stop)] for copy in copies for stop in space.stops(copy)}
    for copy in copies:
        chosen = model.quicksum(z_stop[(copy, stop)]._raw * fires[(copy, stop)] for stop in space.stops(copy))
        model.add_constr(rate >= chosen, name=ctx.unique_name(f"{name}_ge"))
    for bound in sorted(set(fires.values()))[:-1]:
        above = model.quicksum(z_stop[key]._raw for key, value in fires.items() if value > bound)
        model.add_constr(rate <= bound + (1 - bound) * above, name=ctx.unique_name(f"{name}_le"))
    expr = model.quicksum([rate._raw])
    q.add("shared_firing_rate", expr, index=key)
    return expr


def _longest_step(space: DecisionSpace) -> int:
    """The most cycles any node or transfer takes in a slot, on any core or path, or the transfers of a slot keep a link
    they share busy."""
    longest = 0
    for n in space.ssc_nodes:
        longest = max(longest, space.runtime(n))
    busy: dict[tuple[object, int], float] = defaultdict(float)
    for tr in space.transfer_nodes:
        most: dict[object, float] = defaultdict(float)
        for choice in space.path_choices[tr]:
            cycles = space.transfer_latency_for_path(tr, choice)
            longest = max(longest, ceil(cycles))
            for link, share in space.link_load(tr, choice).items():
                if not space.hardware.circuit_switched(link.cores):
                    most[link] = max(most[link], share * cycles)
        for link, cycles in most.items():
            busy[(link, space.slot_of[tr])] += cycles
    return max([longest, *(ceil(cycles) for cycles in busy.values())])
