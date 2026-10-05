"""How much of an iteration the next one overlaps, the fill before the first, and the latency they add up to."""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from math import ceil, prod
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.opt.allocation.constraint_optimization.diagnosis import ConstraintTag
from stream.opt.allocation.constraint_optimization.families import LATENCY, SLOT_PRESSURE, TOTAL_LATENCY
from stream.opt.allocation.constraint_optimization.families.latency import reuse_selectors
from stream.opt.allocation.constraint_optimization.families.routing import LINK_CONTENTION
from stream.opt.allocation.constraint_optimization.space import core_id
from stream.opt.allocation.constraint_optimization.utils import get_active_latency, resource_key
from stream.opt.solver import ObjectiveLevel, SolverVar, SolverVarType
from stream.workload.iterator_type import is_state_operand
from stream.workload.node import ComputationNode

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.mapping.mapping import Resource
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.workload import Tensor, TransferNode

_logger = logging.getLogger(__name__)


class PipeliningModel(Enum):
    """Inter-iteration overlap model: SPAN (idle only before first / after last use) or OCCUPANCY (any unused slot)."""

    SPAN = "span"
    OCCUPANCY = "occupancy"


class Overlap:
    """How much of an iteration the next one overlaps, the fill before the first, and the run's latency, the first
    objective level. ``model`` reclaims any slot a resource leaves unused (``occupancy``) or only those before and
    after its use (``span``); ``transfer_contention`` and ``offchip_contention`` let busy links bound the overlap."""

    name: ClassVar[str] = "overlap"
    requires: ClassVar[tuple[str, ...]] = ("transfer_latency", "reuse_factor", SLOT_PRESSURE)
    provides: ClassVar[tuple[str, ...]] = (
        "overlap",
        "iteration",
        "fill",
        "shared_busy",
        "idle_latency",
        "recurrence_bound",
        TOTAL_LATENCY,
    )

    def __init__(
        self,
        model: str | PipeliningModel = PipeliningModel.OCCUPANCY,
        transfer_contention: bool = True,
        offchip_contention: bool = True,
    ) -> None:
        self.model = PipeliningModel(model)
        self.transfer_contention = transfer_contention
        self.offchip_contention = offchip_contention

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        latency = {key: quantity.expr for key, quantity in q.indexed("transfer_latency").items()}
        idle = _idle_indicators(ctx, self.model)
        idle_lat = _idle_latency_vars(ctx, idle, _slot_pressure_bound(q))
        for res, v in idle_lat.items():
            q.add("idle_latency", v._raw, index=res)
        self._define_overlap_var(ctx, idle_lat, latency)
        _resident_fill(ctx)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q = ctx.quantities
        iterations = ctx.space.iterations
        total_latency = ctx.model.add_var(vtype=SolverVarType.INTEGER, name="total_latency")
        ctx.model.add_constr(
            total_latency
            == iterations * q.get("iteration").expr - (iterations - 1) * q.get("overlap").expr + q.get("fill").expr
        )
        q.add(TOTAL_LATENCY, total_latency._raw)
        return [ObjectiveLevel(expr=total_latency._raw, priority=LATENCY, name="latency")]

    def _define_overlap_var(
        self,
        ctx: FormulationContext,
        idle_lat: dict[Resource, SolverVar],
        latency: dict[tuple[TransferNode, MulticastPathPlan], Any],
    ) -> None:
        q = ctx.quantities
        space, model = ctx.space, ctx.model
        overlap = model.add_var(vtype=SolverVarType.INTEGER, name="overlap")
        iteration = model.quicksum(v._raw for v in ctx.vars.slot_latency.values())
        q.add("overlap", overlap._raw)
        q.add("iteration", iteration)
        busy_terms: dict[CommunicationLink, list[Any]] = defaultdict(list)
        for key in ctx.vars.y:
            for link in space.links_in_choice[key]:
                busy_terms[link].append(latency[key])
        for res, v in idle_lat.items():
            if not self._bounds_overlap(space, res):
                continue
            terms = busy_terms.get(res) if isinstance(res, CommunicationLink) else None
            if terms:
                model.add_constr(overlap <= iteration - model.quicksum(terms))
            else:
                model.add_constr(overlap <= v)
        _skipped_step_floor(ctx, overlap, iteration, latency)
        _shared_bandwidth_bounds(ctx, overlap, iteration)
        rec = _recurrence_bound(space)
        q.add("recurrence_bound", rec)
        if rec > 0:
            model.add_constr(
                overlap <= model.quicksum(v._raw for v in ctx.vars.slot_latency.values()) - rec,
                name="overlap_recurrence_bound",
            )

    def _bounds_overlap(self, space: DecisionSpace, res: Resource) -> bool:
        """Whether this resource being busy is a reason the next iteration cannot start."""
        if isinstance(res, Core):
            return True
        if _is_offchip_link(space, res):
            return self.offchip_contention
        return self.transfer_contention


def _idle_indicators(ctx: FormulationContext, pipelining: PipeliningModel) -> dict[Resource, list[list[SolverVar]]]:
    """Per resource and slot: the binaries whose sum is how much of that slot the next iteration may reclaim."""
    builder = _occupancy_indicators if pipelining is PipeliningModel.OCCUPANCY else _span_indicators
    return {res: builder(ctx, res, active_s, used) for res, active_s, used in _resource_activity(ctx)}


def _resource_activity(ctx: FormulationContext) -> list[tuple[Resource, list[Any], SolverVar]]:
    """Per resource: active per slot and used at all (link = path-choice expr, core = constant)."""
    space, model = ctx.space, ctx.model
    max_s, big_m = space.max_slot, space.big_m
    out: list[tuple[Resource, list[Any], SolverVar]] = []

    carried: dict[CommunicationLink, dict[int, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for key, y in ctx.vars.y.items():
        s = space.slot_of[key[0]]
        for link in space.links_in_choice[key]:
            carried[link][s].append(y._raw)
    for link in space.link_set:
        per_slot = carried[link]
        active_s = [model.quicksum(per_slot.get(s, ())) for s in range(max_s + 1)]
        lu = model.add_var(vtype=SolverVarType.BINARY, name=f"linkUsed_{resource_key(link)}")
        sum_active = model.quicksum(active_s)
        ctx.add_constr(
            sum_active >= lu, name=f"link_used_def_{resource_key(link)}", resource=link, kind=LINK_CONTENTION
        )
        ctx.add_constr(
            sum_active <= big_m * lu, name=f"link_used_def2_{resource_key(link)}", resource=link, kind=LINK_CONTENTION
        )
        out.append((link, active_s, lu))

    core_active_slots: dict[Core, set[int]] = defaultdict(set)
    for node in space.ssc_nodes:
        s = space.slot_of[node]
        for group in space.mapping.get(node).resource_allocation:
            for core in group:
                core_active_slots[core].add(s)
    for core, active_slots in core_active_slots.items():
        lu = model.add_var(vtype=SolverVarType.BINARY, name=f"coreUsed_{resource_key(core)}")
        ctx.add_constr(lu == 1, name=f"core_used_def_{resource_key(core)}", resource=core)
        out.append((core, [1 if s in active_slots else 0 for s in range(max_s + 1)], lu))
    return out


def _span_indicators(ctx: FormulationContext, res: Resource, active_s: list[Any], used: SolverVar) -> list[list[Any]]:
    """SPAN model: only slots before first use and after last use are reclaimable (via prefix/suffix sums)."""
    model, max_s, big_m = ctx.model, ctx.space.max_slot, ctx.space.big_m
    key = resource_key(res)
    prefix = [model.add_var(vtype=SolverVarType.INTEGER, name=f"pre_{key}_{s}") for s in range(max_s + 1)]
    suffix = [model.add_var(vtype=SolverVarType.INTEGER, name=f"suf_{key}_{s}") for s in range(max_s + 1)]
    model.add_constr(prefix[0] == active_s[0])
    model.add_constr(suffix[-1] == active_s[max_s])
    for s in range(1, max_s + 1):
        model.add_constr(prefix[s] == prefix[s - 1] + active_s[s])
        model.add_constr(suffix[max_s - s] == suffix[max_s - s + 1] + active_s[max_s - s])
    indicators = []
    for s in range(max_s + 1):
        is_ = model.add_var(vtype=SolverVarType.BINARY, name=f"idleS_{key}_{s}")
        ie_ = model.add_var(vtype=SolverVarType.BINARY, name=f"idleE_{key}_{s}")
        indicators.append([is_, ie_])
        model.add_constr(prefix[s] <= big_m * (1 - is_))
        model.add_constr(prefix[s] >= used - big_m * is_)
        model.add_constr(suffix[s] <= big_m * (1 - ie_))
        model.add_constr(suffix[s] >= used - big_m * ie_)
        model.add_constr(is_ >= 1 - used)
        model.add_constr(ie_ <= used)
    return indicators


def _occupancy_indicators(
    ctx: FormulationContext, res: Resource, active_s: list[Any], used: SolverVar
) -> list[list[Any]]:
    """OCCUPANCY model: every unused slot is reclaimable -- one indicator per slot, the complement of activity."""
    model, big_m = ctx.model, ctx.space.big_m
    key = resource_key(res)
    indicators = []
    for s in range(ctx.space.max_slot + 1):
        idle = model.add_var(vtype=SolverVarType.BINARY, name=f"idle_{key}_{s}")
        indicators.append([idle])
        act = model.add_var(vtype=SolverVarType.INTEGER, name=f"act_{key}_{s}")
        ctx.add_constr(act == active_s[s], name=f"act_def_{key}_{s}", resource=res)
        ctx.add_constr(act <= big_m * (1 - idle), name=f"idle_off_{key}_{s}", resource=res)
        ctx.add_constr(act >= 1 - idle, name=f"idle_on_{key}_{s}", resource=res)
    return indicators


def _idle_latency_vars(
    ctx: FormulationContext, idle: dict[Resource, list[list[SolverVar]]], slot_latency_ub: int
) -> dict[Resource, SolverVar]:
    """Per resource: the cycles of one iteration it leaves idle for the next to reclaim."""
    model, slot_latency = ctx.model, ctx.vars.slot_latency
    idle_lat: dict[Resource, SolverVar] = {}
    for res in {res for res in idle}:
        key = resource_key(res)
        tag = ConstraintTag(res)
        terms = [
            ctx.binary_scaled_continuous(
                binary_var=ind,
                continuous_var=slot_latency[s],
                continuous_ub=slot_latency_ub,
                base_name=f"idle{i}_lat_{key}_{s}",
                tag=tag,
            )
            for s, indicators in enumerate(idle[res])
            for i, ind in enumerate(indicators)
        ]
        v = model.add_var(vtype=SolverVarType.INTEGER, name=f"idleLat_{key}")
        ctx.add_constr(v == model.quicksum(t._raw for t in terms), name=f"idleLat_def_{key}", resource=res)
        idle_lat[res] = v
    return idle_lat


def _slot_pressure_bound(q: QuantityRegistry) -> int:
    """The largest slot latency the families' constraints can force, a safe bound on every slot."""
    pressures = q.indexed(SLOT_PRESSURE)
    if unbounded := [index for index, p in pressures.items() if p.upper_bound is None]:
        raise ValueError(f"{SLOT_PRESSURE} quantities need an upper_bound: {unbounded}")
    return ceil(max(p.upper_bound for p in pressures.values()))


def _skipped_step_floor(
    ctx: FormulationContext, overlap: SolverVar, iteration: Any, latency: dict[tuple[Any, Any], Any]
) -> None:
    """A step a core skips still takes as long as the operands it skips over."""
    space = ctx.space
    for n in space.ssc_nodes:
        for core in space.cost_lut.get_cores(n):
            entry = space.cost_lut.get_cost(n, core)
            fraction = (entry.metadata or {}).get("computed_fraction", 1.0)
            if fraction >= 1.0 - 1e-9:
                continue
            busy = get_active_latency(n, float(ceil(entry.latency_total)), space.ssis)
            for key in ctx.vars.y:
                if core not in space.choice_src_cores[key] and core not in space.choice_dst_cores[key]:
                    continue
                tr, choice = key
                ctx.add_constr(
                    overlap <= iteration - busy - (1.0 - fraction) * latency[key],
                    name=f"skip_floor_{n.name}_{resource_key(core)}_{tr.name}_{hash(choice)}",
                    resource=core,
                )


def _is_offchip_link(space: DecisionSpace, res: Resource) -> bool:
    """A link with the off-chip core at either end."""
    off = space.offchip_core_id
    if off is None or not isinstance(res, CommunicationLink):
        return False
    return off in (core_id(res.sender), core_id(res.receiver))


def _recurrence_bound(space: DecisionSpace) -> int:
    """Cycles a loop-carried state forbids overlapping (modulo scheduling's RecMII), 0 when feed-forward: every
    state is a distance-one self-loop on the node that keeps it, so the worst cycle is the slowest carrier alone."""
    carriers = [n for n in space.ssc_nodes if any(is_state_operand(n, t) for t in n.inputs)]
    return max((space.runtime(n) for n in carriers), default=0)


def _shared_bandwidth_bounds(ctx: FormulationContext, overlap: SolverVar, iteration: Any) -> None:
    """Bound the step by the time each shared-bandwidth core spends on one iteration's transfers."""
    q = ctx.quantities
    space, model = ctx.space, ctx.model
    for core in space.shared_bandwidth:
        terms = [
            _active_shared_latency(ctx, core, tr, choice, y)._raw
            for (tr, choice), y in ctx.vars.y.items()
            if space.direction(core, choice) is not None and not space.choice_shares_memory(tr, choice)
        ]
        if not terms:
            continue
        busy = model.add_var(vtype=SolverVarType.CONTINUOUS, name=f"shared_busy_{core}")
        model.add_constr(busy == model.quicksum(terms), name=f"shared_busy_{core}_def")
        model.add_constr(overlap <= iteration - busy, name=f"shared_bound_{core}")
        q.add("shared_busy", busy._raw, index=core)


def _active_shared_latency(
    ctx: FormulationContext, core: int, tr: TransferNode, choice: MulticastPathPlan, y: SolverVar
) -> SolverVar:
    """Cycles this transfer holds a shared-bandwidth core per iteration, at its read and write ceiling."""
    q = ctx.quantities
    space = ctx.space
    constant = float(space.shared_cycles(core, tr, choice, space.shared_bandwidth[core].ceiling))
    return ctx.binary_times_const_over_linexpr(
        binary_var=y,
        numerator=get_active_latency(tr, constant, space.ssis),
        denominator_expr=q.get("reuse_factor", tr).expr,
        denominator_lb=1.0,
        base_name=f"shared_latency_{core}_{tr}",
        selectors=reuse_selectors(ctx, tr.outputs[0]),
    )


def _resident_fill(ctx: FormulationContext) -> None:
    """Cycles each run waits for the off-chip windows it holds in one buffer to fill: such a window fills
    before the iteration that reads it, overlapping none. The fills share each path and shared-bandwidth core
    as one iteration's transfers do."""
    q = ctx.quantities
    space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
    fill = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name="resident_fill")
    shared: dict[int, list[Any]] = defaultdict(list)
    singles: dict[Tensor, list[tuple[int, Any]]] = defaultdict(list)
    for (t, s), single in ctx.vars.z_single.items():
        singles[t].append((s, single))
    optimized = set(space.tensors_to_optimize_reuse_for)
    for (tr, choice), y in ctx.vars.y.items():
        t = tr.outputs[0]
        if t not in optimized:
            continue
        sizes = space.ssis[t].get_applicable_temporal_sizes()
        whole = (
            [
                (space.tiles_needed_levels[(t, s)], z_stop[(t, s)], f"L{s}")
                for s in range(len(sizes))
                if not space.rotation_levels[(t, s)]
            ]
            if space.is_const_i(tr)
            else []
        )
        choices = whole + [(prod(sizes[s + 1 :]), single, f"single_L{s}") for s, single in singles[t]]
        held = [
            (tiles, ctx.binary_product(a=y, b=z, base_name=f"fill_{tr.name}_{hash(choice)}_{name}"))
            for tiles, z, name in choices
        ]
        if not held:
            continue
        cycles = space.transfer_latency_for_path(tr, choice)
        model.add_constr(
            fill >= model.quicksum(tiles * cycles * w._raw for tiles, w in held),
            name=f"fill_{tr.name}_{hash(choice)}",
        )
        for core, bandwidth in space.shared_bandwidth.items():
            share = space.shared_cycles(core, tr, choice, bandwidth.ceiling)
            shared[core] += [tiles * share * w._raw for tiles, w in held]
    for core, terms in shared.items():
        model.add_constr(fill >= model.quicksum(terms), name=f"fill_shared_{core}")
    q.add("fill", fill._raw + warmup if (warmup := _warmup(ctx)) is not None else fill._raw)


def _warmup(ctx: FormulationContext) -> Any:
    """Cycles the first iteration adds where a sliding window's first tile is longer than its interior one, priced at
    the interior cost per element: a transfer delays its reader, a node the readers on cores it does not run on, and
    the shorter last tiles end behind the sink's interior one; None where no window slides."""
    space, model = ctx.space, ctx.model
    if not space.warmup:
        return None
    cost: dict[Any, Any] = {n: space.active_runtime(n) for n in space.ssc_nodes}
    for (tr, _), quantity in ctx.quantities.indexed("transfer_latency").items():
        cost[tr] = cost.get(tr, 0) + quantity.expr
    ready = {n: model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name=f"ready_{n.name}") for n in cost}
    warmup = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name="warmup")
    for node, start in ready.items():
        delay = space.warmup.get(node, 0.0) * cost[node]
        readers = [c for c in space.workload.successors(node) if c in ready]
        waits = not isinstance(node, ComputationNode) or space.runs_readers_elsewhere(node)
        for reader in readers:
            model.add_constr(ready[reader] >= start + (delay if waits else 0), name=f"warmup_{node.name}_{reader.name}")
        if not readers:
            model.add_constr(warmup >= start + delay, name=f"warmup_{node.name}")
    return warmup._raw
