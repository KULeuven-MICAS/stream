"""The bits crossing the off-chip boundary each iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from stream.opt.allocation.constraint_optimization.families import LATENCY, OFFCHIP_TIMED, OFFCHIP_TRAFFIC
from stream.opt.allocation.constraint_optimization.families.latency import is_single_copy, reuse_selectors
from stream.opt.allocation.constraint_optimization.utils import active_fraction
from stream.opt.solver import ObjectiveLevel, SolverVarType

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import CopySet, DecisionSpace
    from stream.opt.solver import SolverVar
    from stream.workload.workload import TransferNode


class OffchipTraffic:
    """The bits crossing the off-chip boundary, the objective level after the latency; with ``charge`` the latency
    also pays the time the off-chip links take to move them, unless a shared-bandwidth model of each off-chip memory
    or the overlap's off-chip contention already times them; several off-chip memories need the latter."""

    name: ClassVar[str] = "offchip_traffic"
    requires: ClassVar[tuple[str, ...]] = ("transfer_latency", "copy_firing_rate")
    provides: ClassVar[tuple[str, ...]] = ("offchip_traffic_weight",)

    def __init__(self, charge: bool = True) -> None:
        self.charge = charge

    def build(self, ctx: FormulationContext) -> None:
        space = ctx.space
        timed = all(memory in space.shared_bandwidth for memory in space.accelerator.offchip_memories)
        bandwidth = space.offchip_bandwidth()
        if self.charge and not timed and bandwidth:
            ctx.quantities.add("offchip_traffic_weight", space.iterations / bandwidth)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q, space, z_stop = ctx.quantities, ctx.space, ctx.vars.z_stop
        traffic = ctx.model.quicksum(
            [
                *(
                    space.copied_bits(t) / space.reuse_levels[(t, s)] * z_stop[(t, s)]._raw
                    for t in space.tensors_to_optimize_reuse_for
                    for s in space.stops(t)
                ),
                *_duplicate_offchip_terms(ctx),
            ]
        )._raw
        levels = [ObjectiveLevel(expr=traffic, priority=OFFCHIP_TRAFFIC, name="offchip_traffic")]
        if "offchip_traffic_weight" in q and OFFCHIP_TIMED not in q:
            if len(offchip := space.accelerator.offchip_memories) > 1:
                raise ValueError(
                    f"off-chip memories {[memory.name for memory in offchip]}: the overlap family with "
                    "offchip_contention times the links of each, and this selection leaves it off"
                )
            levels.append(
                ObjectiveLevel(expr=q.get("offchip_traffic_weight").expr * traffic, priority=LATENCY, name="latency")
            )
        return levels


def duplicate_offchip_bits(
    space: DecisionSpace, tr: TransferNode, choice: MulticastPathPlan, copies: CopySet | None
) -> float:
    """Bits one firing of ``tr`` on ``choice`` off-chip memory sends or receives more than once for ``copies`` (all if
    None): one duplicate per extra link a multicast leaves on."""
    repeats = sum(
        max(0.0, space.memory_share(tr, choice, m, True, copies) - choice.unique_sent_share(m, copies))
        for m in dict.fromkeys(choice.sources)
        if m.offchip and choice.links_at(m, leaving=True)
    )
    repeats += sum(
        max(0.0, space.memory_share(tr, choice, m, False, copies) - 1)
        for m in dict.fromkeys(choice.targets)
        if m.offchip
    )
    return repeats * space.moved_bits(tr, choice, copies=copies) if repeats else 0.0


def _duplicate_offchip_terms(ctx: FormulationContext) -> list[Any]:
    """Per transfer, the :func:`duplicate_offchip_bits` of the route it takes per iteration: at the copy's rate, or for
    routes delivering to one copy, per reuse stop of the copy (:func:`_at_stop`)."""
    q, space = ctx.quantities, ctx.space
    terms = []
    single: dict[TransferNode, dict[MulticastPathPlan, float]] = {}
    for tr, choice in q.indexed("transfer_latency") if "transfer_latency" in q else {}:
        if is_single_copy(groups := space.copy_sets(tr, choice)):
            if bits := duplicate_offchip_bits(space, tr, choice, None):
                single.setdefault(tr, {})[choice] = bits
            continue
        for copies in groups:
            if bits := duplicate_offchip_bits(space, tr, choice, copies):
                rate = q.get("copy_firing_rate", (tr, choice, copies)).expr
                terms.append(bits / active_fraction(tr, space.ssis) * rate)
    for tr, routes in single.items():
        groups = space.copy_sets(tr, next(iter(routes)))
        copy = tr.outputs[groups[0][0]] if groups else tr.outputs[0]
        terms += [_at_stop(ctx, tr, routes, z) / reuse for z, reuse in reuse_selectors(ctx, copy)]
    return terms


def _at_stop(ctx: FormulationContext, tr: TransferNode, routes: dict[MulticastPathPlan, float], z: SolverVar) -> Any:
    """A variable at least the bits of the route of ``routes`` ``tr`` takes where the stop ``z`` is chosen."""
    y = ctx.vars.y
    chosen = ctx.model.quicksum(bits * y[(tr, route)]._raw for route, bits in routes.items())._raw
    most = max(routes.values())
    bits = ctx.model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=most, name=ctx.unique_name("duplicate_offchip"))
    ctx.model.add_constr(bits >= chosen - most * (1 - z), name=ctx.unique_name("duplicate_offchip_lb"))
    return bits._raw
