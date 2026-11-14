"""How much of an iteration the next one overlaps, the fill before the first, and the latency they add up to."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from math import ceil, prod
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.opt.allocation.constraint_optimization.diagnosis import ConstraintTag, HardwareResource
from stream.opt.allocation.constraint_optimization.families import (
    LATENCY,
    LATENCY_WITHOUT_WAIT,
    OFFCHIP_TIMED,
    SLOT_PRESSURE,
    TOTAL_LATENCY,
)
from stream.opt.allocation.constraint_optimization.families.latency import (
    link_busy,
    per_copy_terms,
    single_copy,
    steady_state_cycles,
)
from stream.opt.allocation.constraint_optimization.families.routing import LINK_CONTENTION
from stream.opt.allocation.constraint_optimization.families.traffic import call_port_cycles, firing_port_cycles
from stream.opt.allocation.constraint_optimization.utils import get_active_latency, resource_key
from stream.opt.solver import LinExpr, ObjectiveLevel, SolverVar, SolverVarType
from stream.workload.demand import DemandItem, firing_count, moved_elements
from stream.workload.iterator_type import is_state_operand
from stream.workload.node import ComputationNode

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
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
    requires: ClassVar[tuple[str, ...]] = ("transfer_latency", "copy_reuse_factor", SLOT_PRESSURE)
    provides: ClassVar[tuple[str, ...]] = (
        "overlap",
        "iteration",
        "fill",
        "shared_busy",
        "idle_latency",
        "recurrence_bound",
        "absent_wait",
        "epilogue",
        TOTAL_LATENCY,
        OFFCHIP_TIMED,
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
        self.has_absent_wait = False
        self.has_epilogue = False

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        latency = {key: quantity.expr for key, quantity in q.indexed("transfer_latency").items()}
        idle = _idle_indicators(ctx, self.model)
        idle_lat = _idle_latency_vars(ctx, idle, _slot_pressure_bound(q))
        for res, v in idle_lat.items():
            q.add("idle_latency", v._raw, index=res)
        self._define_overlap_var(ctx, idle_lat, latency)
        drained = _drained(ctx.space)
        fill = _resident_fill(ctx, max(drained.values(), default=0))
        epilogue = _epilogue(ctx, drained)
        self.has_epilogue = epilogue is not None
        q.add("epilogue", epilogue if self.has_epilogue else 0)
        fill = fill + epilogue if self.has_epilogue else fill
        wait = _absent_wait(ctx, fill, drained)
        self.has_absent_wait = wait is not None
        q.add("absent_wait", wait if self.has_absent_wait else 0)
        q.add("fill", fill + wait if self.has_absent_wait else fill)
        if self.offchip_contention:
            q.add(OFFCHIP_TIMED, 1)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q = ctx.quantities
        iterations = ctx.space.iterations
        total_latency = ctx.model.add_var(vtype=SolverVarType.CONTINUOUS, name="total_latency")
        ctx.model.add_constr(
            total_latency
            == iterations * q.get("iteration").expr - (iterations - 1) * q.get("overlap").expr + q.get("fill").expr
        )
        q.add(TOTAL_LATENCY, total_latency._raw)
        levels = [ObjectiveLevel(expr=total_latency._raw, priority=LATENCY, name="latency")]
        if not (self.has_absent_wait or self.has_epilogue):
            return levels
        without_wait = total_latency._raw - q.get("absent_wait").expr - q.get("epilogue").expr
        return [*levels, ObjectiveLevel(expr=without_wait, priority=LATENCY_WITHOUT_WAIT, name="latency_without_wait")]

    def _define_overlap_var(
        self,
        ctx: FormulationContext,
        idle_lat: dict[HardwareResource, SolverVar],
        latency: dict[tuple[TransferNode, MulticastPathPlan], Any],
    ) -> None:
        q = ctx.quantities
        space, model = ctx.space, ctx.model
        overlap = model.add_var(vtype=SolverVarType.INTEGER, name="overlap")
        iteration = model.quicksum(v._raw for v in ctx.vars.slot_latency.values())
        q.add("overlap", overlap._raw)
        q.add("iteration", iteration)
        busy_terms: dict[CommunicationLink, list[Any]] = defaultdict(list)
        for (link, _), terms in link_busy(ctx).items():
            busy_terms[link] += terms
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

    def _bounds_overlap(self, space: DecisionSpace, res: HardwareResource) -> bool:
        """Whether this resource being busy is a reason the next iteration cannot start."""
        if isinstance(res, Core):
            return True
        if _is_offchip_link(space, res):
            return self.offchip_contention
        return self.transfer_contention


def _idle_indicators(
    ctx: FormulationContext, pipelining: PipeliningModel
) -> dict[HardwareResource, list[list[SolverVar]]]:
    """Per resource and slot: the binaries whose sum is how much of that slot the next iteration may reclaim."""
    builder = _occupancy_indicators if pipelining is PipeliningModel.OCCUPANCY else _span_indicators
    return {res: builder(ctx, res, active_s, used) for res, active_s, used in _resource_activity(ctx)}


def _resource_activity(ctx: FormulationContext) -> list[tuple[HardwareResource, list[Any], SolverVar]]:
    """Per resource: active per slot and used at all (link = path-choice expr, core = constant)."""
    space, model = ctx.space, ctx.model
    max_s, big_m = space.max_slot, space.big_m
    out: list[tuple[HardwareResource, list[Any], SolverVar]] = []

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


def _span_indicators(
    ctx: FormulationContext, res: HardwareResource, active_s: list[Any], used: SolverVar
) -> list[list[Any]]:
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
    ctx: FormulationContext, res: HardwareResource, active_s: list[Any], used: SolverVar
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
    ctx: FormulationContext, idle: dict[HardwareResource, list[list[SolverVar]]], slot_latency_ub: int
) -> dict[HardwareResource, SolverVar]:
    """Per resource: the cycles of one iteration it leaves idle for the next to reclaim."""
    model, slot_latency = ctx.model, ctx.vars.slot_latency
    idle_lat: dict[HardwareResource, SolverVar] = {}
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
    """A step a core skips still takes as long as the operands it skips over, for each transfer whose producer or
    reader runs on that core."""
    space = ctx.space
    for n in space.ssc_nodes:
        for core in space.cost_lut.get_cores(n):
            entry = space.cost_lut.get_cost(n, core)
            fraction = (entry.metadata or {}).get("computed_fraction", 1.0)
            if fraction >= 1.0 - 1e-9:
                continue
            busy = get_active_latency(n, float(ceil(entry.latency_total)), space.ssis)
            for key in ctx.vars.y:
                tr, choice = key
                if core not in space.compute_cores(tr):
                    continue
                ctx.add_constr(
                    overlap <= iteration - busy - (1.0 - fraction) * latency[key],
                    name=f"skip_floor_{n.name}_{resource_key(core)}_{space.route_name(tr, choice)}",
                    resource=core,
                )


def _is_offchip_link(space: DecisionSpace, res: HardwareResource) -> bool:
    """A link with the off-chip memory at either end."""
    return isinstance(res, CommunicationLink) and any(memory.offchip for memory in res.memories)


def _recurrence_bound(space: DecisionSpace) -> int:
    """Cycles a loop-carried state forbids overlapping (modulo scheduling's RecMII), 0 when feed-forward: every
    state is a distance-one self-loop on the node that keeps it, so the worst cycle is the slowest carrier alone."""
    carriers = [n for n in space.ssc_nodes if any(is_state_operand(n, t) for t in n.inputs)]
    return max((space.runtime(n) for n in carriers), default=0)


def _shared_bandwidth_bounds(ctx: FormulationContext, overlap: SolverVar, iteration: Any) -> None:
    """Bound the step by the time each shared-bandwidth memory spends on one iteration's transfers."""
    q = ctx.quantities
    space, model = ctx.space, ctx.model
    for memory in space.shared_bandwidth:
        terms = [
            _active_shared_latency(ctx, memory, tr, choice, y)._raw
            for (tr, choice), y in ctx.vars.y.items()
            if space.direction(memory, choice) is not None and not space.choice_shares_memory(tr, choice)
        ]
        if not terms:
            continue
        busy = model.add_var(vtype=SolverVarType.CONTINUOUS, name=f"shared_busy_{memory.name}")
        model.add_constr(busy == model.quicksum(terms), name=f"shared_busy_{memory.name}_def")
        model.add_constr(overlap <= iteration - busy, name=f"shared_bound_{memory.name}")
        q.add("shared_busy", busy._raw, index=memory)


def _active_shared_latency(
    ctx: FormulationContext, memory: Memory, tr: TransferNode, choice: MulticastPathPlan, y: SolverVar
) -> SolverVar | LinExpr:
    """Cycles this transfer holds a shared-bandwidth memory per iteration, at its read and write ceiling."""
    space = ctx.space
    ceiling = space.shared_bandwidth[memory].ceiling
    terms = per_copy_terms(
        ctx,
        tr,
        choice,
        y,
        lambda copies: float(space.shared_cycles(memory, tr, choice, ceiling, copies)),
        f"shared_latency_{memory.name}_{tr}",
    )
    if None in terms and space.once_per_run_stops(tr, single_copy(ctx, tr, choice)):
        return steady_state_cycles(
            ctx, tr, choice, terms[None], float(space.shared_cycles(memory, tr, choice, ceiling))
        )
    return next(iter(terms.values())) if len(terms) == 1 else ctx.model.quicksum(terms.values())


def _drained(space: DecisionSpace) -> dict[object, int]:
    """Per node, the cycles the last iteration's results take to leave the systolic arrays up to it, which no later
    tile overlaps: a node reading another's results waits for them to drain, while one that does not streams into its
    array behind it, so the run waits for the drains along its longest chain of dependent nodes."""
    drained: dict[object, int] = {}
    for n in space.workload.dataflow_sort():
        own = 0
        if isinstance(n, ComputationNode):
            entries = (space.cost_lut.get_cost(n, core) for core in space.cost_lut.get_cores(n))
            own = max((ceil((entry.metadata or {}).get("drain_cycles", 0)) for entry in entries), default=0)
        drained[n] = max((drained[p] for p in space.workload.predecessors(n)), default=0) + own
    return drained


def _resident_fill(ctx: FormulationContext, drain: int = 0) -> Any:
    """Cycles each run waits for the off-chip windows it holds in one buffer to fill: such a window fills
    before the iteration that reads it, overlapping none. The fills share each path and shared-bandwidth memory
    as one iteration's transfers do. ``drain`` adds the cycles the last iteration's results take to leave the
    systolic arrays."""
    space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
    fill = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name="resident_fill")
    shared: dict[Memory, list[Any]] = defaultdict(list)
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
        route = space.route_name(tr, choice)
        held = [(tiles, ctx.binary_product(a=y, b=z, base_name=f"fill_{route}_{name}")) for tiles, z, name in choices]
        if not held:
            continue
        cycles = space.transfer_latency_for_path(tr, choice)
        model.add_constr(
            fill >= model.quicksum(tiles * cycles * w._raw for tiles, w in held),
            name=f"fill_{route}",
        )
        for memory, bandwidth in space.shared_bandwidth.items():
            share = space.shared_cycles(memory, tr, choice, bandwidth.ceiling)
            shared[memory] += [tiles * share * w._raw for tiles, w in held]
    for memory, terms in shared.items():
        model.add_constr(fill >= model.quicksum(terms), name=f"fill_shared_{memory.name}")
    fill_cycles = fill._raw + drain
    return fill_cycles + warmup if (warmup := _warmup(ctx)) is not None else fill_cycles


def _epilogue(ctx: FormulationContext, drained: dict[object, int]) -> Any:
    """The epilogue: cycles the run lasts after its last iteration and drain, for the last firing of a copy kept along a
    loop its producer still computes in (:meth:`DecisionSpace.epilogue_stops`) and every call and firing after it. None
    where there is none."""
    space, model, q = ctx.space, ctx.model, ctx.quantities
    nodes = space.epilogue_nodes
    if not nodes:
        return None
    drain = max(drained.values(), default=0)
    after = q.get("iteration").expr + drain
    slot_bound = _slot_pressure_bound(q)
    epilogue = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name="epilogue")
    calls = call_port_cycles(ctx)
    done: dict[Any, SolverVar] = {}
    in_epilogue: dict[Any, SolverVar] = {}
    for node in nodes:
        done[node] = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name=f"epilogue_done_{node.name}")
        producers = [p for p in space.workload.predecessors(node) if p in space.slot_of]
        in_epilogue[node] = _in_epilogue(ctx, node, producers, in_epilogue)
        cost = _epilogue_cost(ctx, node, in_epilogue[node], producers, drained, calls)
        if node not in space.ssc_nodes:
            reach = (space.slot_of[node] + 1) * slot_bound + drained[node]
            model.add_constr(
                done[node] >= _slot_end(ctx, node, drained) - reach * in_epilogue[node],
                name=ctx.unique_name(f"epilogue_slot_{node.name}"),
            )
        for k, p in enumerate(producers):
            ready = done[p] if p in done else _slot_end(ctx, p, drained)
            model.add_constr(done[node] >= ready + cost, name=ctx.unique_name(f"epilogue_{node.name}_{k}"))
        model.add_constr(epilogue >= done[node] - after, name=ctx.unique_name(f"epilogue_end_{node.name}"))
    return epilogue._raw


def _in_epilogue(
    ctx: FormulationContext, node: Any, producers: list[Any], in_epilogue: dict[Any, SolverVar]
) -> SolverVar:
    """1 where ``node`` runs in the epilogue, else 0: where it :meth:`DecisionSpace.runs_in_epilogue`,
    a copy of it is held at a :meth:`DecisionSpace.epilogue_stops` stop, or a producer does."""
    space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
    once = node in space.ssc_nodes and space.runs_in_epilogue(node)
    var = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=float(once), ub=1.0, name=f"in_epilogue_{node.name}")
    if once:
        return var
    causes = [in_epilogue[p]._raw for p in producers if p in in_epilogue]
    if node not in space.ssc_nodes:
        causes += [
            model.quicksum(z_stop[(copy, s)]._raw for s in sorted(stops))
            for copy in node.outputs
            if (stops := space.epilogue_stops(node, copy))
        ]
    for cause in causes:
        model.add_constr(var >= cause, name=ctx.unique_name(f"in_epilogue_{node.name}"))
    model.add_constr(var <= model.quicksum(causes), name=ctx.unique_name(f"in_epilogue_{node.name}_ub"))
    return var


def _slot_end(ctx: FormulationContext, node: Any, drained: dict[object, int]) -> Any:
    """Cycles into the last iteration ``node``'s slot ends, its drain added."""
    slots = ctx.vars.slot_latency
    return ctx.model.quicksum([*(slots[s]._raw for s in range(ctx.space.slot_of[node] + 1)), drained[node]])


def _epilogue_cost(  # noqa: PLR0913
    ctx: FormulationContext,
    node: Any,
    in_epilogue: SolverVar,
    producers: list[Any],
    drained: dict[object, int],
    calls: dict[ComputationNode, float],
) -> Any:
    """Cycles ``node`` adds to the epilogue where it runs in it (``in_epilogue``): a transfer's whole firing on the
    route it takes, or a computation node's whole call, at its busiest port if that is longer (``calls``), and its own
    drain."""
    space, model = ctx.space, ctx.model
    if node in space.ssc_nodes:
        own = drained[node] - max((drained[p] for p in producers), default=0)
        return (max(space.runtime(node), calls.get(node, 0.0)) + own) * in_epilogue
    firings = {choice: _firing_cycles(ctx, node, choice) for choice in space.path_choices[node]}
    longest = max(firings.values(), default=0)
    chosen = model.quicksum(cycles * ctx.vars.y[(node, choice)]._raw for choice, cycles in firings.items())
    return chosen + longest * in_epilogue - longest


def _firing_cycles(ctx: FormulationContext, tr: TransferNode, choice: MulticastPathPlan) -> float:
    """Cycles one whole firing of ``tr`` on ``choice`` holds its busiest resource: its links, a top-level port or a
    shared-bandwidth memory at its ceiling."""
    space = ctx.space
    cycles = [float(space.transfer_latency_for_path(tr, choice)), firing_port_cycles(ctx, tr, choice)]
    if not space.choice_shares_memory(tr, choice):
        cycles += [
            float(space.shared_cycles(memory, tr, choice, bandwidth.ceiling))
            for memory, bandwidth in space.shared_bandwidth.items()
            if space.direction(memory, choice) is not None
        ]
    return max(cycles)


def _warmup(ctx: FormulationContext) -> Any:
    """Cycles a sliding window's longer first tiles add, at the interior cost per element: a node's longer first and
    shorter last tiles cancel on its own resources, so what counts is who waits on them, a transfer's reader and the
    readers of a node behind a moving transfer or on other cores; None where no window slides."""
    space, model = ctx.space, ctx.model
    if not space.warmup:
        return None
    cost = _cycles_per_iteration(ctx)
    ahead = _transfer_run_ahead(ctx)
    ready = {n: model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name=f"ready_{n.name}") for n in cost}
    warmup = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, name="warmup")
    for node, start in ready.items():
        waits = not isinstance(node, ComputationNode) or space.is_waited_on(node)
        delay = ahead.get(node, space.warmup.get(node, 0.0) * cost[node]) if waits else 0
        for reader in (c for c in space.workload.successors(node) if c in ready):
            model.add_constr(ready[reader] >= start + delay, name=f"warmup_{node.name}_{reader.name}")
        model.add_constr(warmup >= start, name=f"warmup_{node.name}")
    return warmup._raw


def _transfer_run_ahead(ctx: FormulationContext) -> dict[TransferNode, Any]:
    """Cycles each transfer runs ahead of its steady pace at most: the sliding work's lookahead at the average
    firing's cost, or where the route counts each firing, the most its first firings take beyond the average."""
    space = ctx.space
    ahead: dict[TransferNode, Any] = {}
    for (tr, choice), quantity in ctx.quantities.indexed("transfer_latency").items():
        excess = _first_firings_excess(choice.delivered_items)
        share = space.warmup.get(tr, 0.0) if excess is None else excess
        ahead[tr] = ahead.get(tr, 0) + share * quantity.expr
    return ahead


def _first_firings_excess(items: Sequence[DemandItem]) -> float | None:
    """The most the first firings carry beyond as many average firings, in average firings; None where one firing
    stands for all."""
    if firing_count(items) <= 1:
        return None
    average = moved_elements(items)
    if not average:
        return 0.0
    spans: dict[tuple[int, int], list[DemandItem]] = {}
    for item in items:
        spans.setdefault(item.box[0], []).append(replace(item, box=item.box[1:], firings=1))
    running, most = 0.0, 0.0
    for (lo, hi), group in sorted(spans.items()):
        running += (hi - lo + 1) * (moved_elements(group) - average)
        most = max(most, running)
    return most / average


def _cycles_per_iteration(ctx: FormulationContext) -> dict[Any, Any]:
    """Cycles each node and transfer takes per iteration."""
    cost: dict[Any, Any] = {n: ctx.space.active_runtime(n) for n in ctx.space.ssc_nodes}
    for (tr, _), quantity in ctx.quantities.indexed("transfer_latency").items():
        cost[tr] = cost.get(tr, 0) + quantity.expr
    return cost


def _absent_wait(ctx: FormulationContext, fill: Any, drained: dict[object, int]) -> Any:
    """Cycles the run lasts beyond its iterations and ``fill`` for the whole calls along the chains through
    :attr:`DecisionSpace.waiting_absent_nodes`, each node draining ``drained`` cycles; None where no node waits."""
    space, model, q = ctx.space, ctx.model, ctx.quantities
    chain = _chain(space)
    if not chain:
        return None
    wait = model.add_var(vtype=SolverVarType.CONTINUOUS, name="absent_wait")
    iterations = space.iterations
    steady = iterations * q.get("iteration").expr - (iterations - 1) * q.get("overlap").expr + fill
    step = _cycles_per_iteration(ctx)
    call: dict[Any, Any] = {n: space.runtime(n) for n in space.ssc_nodes}
    for (tr, choice), y in ctx.vars.y.items():
        call[tr] = call.get(tr, 0) + space.transfer_latency_for_path(tr, choice) * y._raw
    first = {n: model.add_var(vtype=SolverVarType.CONTINUOUS, name=f"first_call_{n.name}") for n in chain}
    last = {n: model.add_var(vtype=SolverVarType.CONTINUOUS, name=f"last_call_{n.name}") for n in chain}
    for node in chain:
        model.add_constr(last[node] >= first[node] + iterations * step[node], name=f"calls_{node.name}")
        for reader in (c for c in space.workload.successors(node) if c in first):
            waited = space.calls_before_first_read(node, reader) * call[node]
            model.add_constr(first[reader] >= first[node] + waited, name=f"first_call_{node.name}_{reader.name}")
            model.add_constr(last[reader] >= last[node] + call[reader], name=f"last_call_{node.name}_{reader.name}")
        model.add_constr(wait >= last[node] + drained.get(node, 0) - steady, name=f"absent_wait_{node.name}")
    return wait._raw


def _chain(space: DecisionSpace) -> list[Any]:
    """The nodes and transfers upstream or downstream of a waiting absent node, in dataflow order."""
    nodes = {*space.ssc_nodes, *space.transfer_nodes}
    found: set[Any] = set(space.waiting_absent_nodes)
    for neighbours in (space.workload.predecessors, space.workload.successors):
        frontier = list(space.waiting_absent_nodes)
        while frontier:
            for n in neighbours(frontier.pop()):
                if n in nodes and n not in found:
                    found.add(n)
                    frontier.append(n)
    return [n for n in space.workload.topological_sort() if n in found]
