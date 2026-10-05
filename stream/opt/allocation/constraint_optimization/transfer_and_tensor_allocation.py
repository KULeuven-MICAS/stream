import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from math import ceil
from typing import Any, TypeAlias

from stream.allocation.problem import SteadyStateProblem
from stream.allocation.solution import AllocationSolution, Latency
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.diagnosis import (
    ConstraintTag,
    StructuralRule,
    infeasibility_report,
)
from stream.opt.allocation.constraint_optimization.families import (
    FamilySelection,
    ObjectiveFamily,
    ScreeningFamily,
)
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.report import solved_reports
from stream.opt.allocation.constraint_optimization.space import DecisionSpace, Placement
from stream.opt.allocation.constraint_optimization.timeslot_allocation import _resource_key
from stream.opt.allocation.constraint_optimization.utils import get_active_latency
from stream.opt.solver import (
    ObjectiveLevel,
    SolverBackend,
    SolverModel,
    SolverParams,
    SolverVar,
    SolverVarType,
    create_solver,
)
from stream.profiling import span
from stream.workload.steady_state.iteration_space import Reuse
from stream.workload.workload import Tensor, TransferNode

_logger = logging.getLogger(__name__)

REUSE_CHOICE = StructuralRule(
    "No consistent reuse schedule",
    "A tensor's reuse level (how long it stays resident) has no value consistent with the other pinned tensors "
    "-- typically a streamed (K-tiled) weight competing with a pinned activation for the same on-chip budget.",
)

TensorReuseLevels: TypeAlias = dict[Tensor, int]
TensorDepths: TypeAlias = dict[Tensor, int]
TensorAlloc: TypeAlias = dict[Tensor, Placement]
TransferAlloc: TypeAlias = dict[TransferNode, MulticastPathPlan]
MemoryAlloc: TypeAlias = dict[TransferNode, Placement]


class TransferAndTensorAllocator:
    """The allocation model of a steady-state problem: the core decision variables -- where every movable tensor
    lives, which route each transfer takes, where each tensor's reuse stops -- the constraints and objective levels
    its families build on them, and the allocation its solve reads back."""

    VAR_THRESHOLD = 0.5

    def __init__(
        self,
        problem: SteadyStateProblem,
        *,
        families: FamilySelection,
        backend: str = "ORTOOLS_GSCIP",
    ):
        self.families = families
        self.space = DecisionSpace(problem)
        self.model: SolverModel = create_solver(SolverBackend[backend], "transfer_tensor_alloc")
        self.model.set_param(SolverParams.VERBOSITY, 1)
        self.model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
        self.quantities = QuantityRegistry()
        self.ledger = ResourceLedger()
        self._build_model()

    def _build_model(self) -> None:
        with span("variables"):
            self.vars = self._create_variables()
        self.context = FormulationContext(self.space, self.vars, self.model, self.quantities, self.ledger)
        with span("capacity_screen"):
            for family in self.families.families:
                if isinstance(family, ScreeningFamily):
                    family.screen(self.context)
        for name, build in self.families.steps:
            with span(f"family_{name}"):
                build(self.context, self.quantities)
        with span("objective"):
            self.objective = self._objective_levels()
            self.model.set_lexicographic_objectives(list(self.objective.values()), sense="minimize")

    def _create_variables(self) -> DecisionVariables:
        space, model = self.space, self.model
        x: dict[tuple[Tensor, Placement], SolverVar] = {}
        for t in space.tensor_var:
            for choice in space.tensor_choices[t]:
                choice_name = "__".join(_resource_key(c) for c in choice)
                x[(t, choice)] = model.add_var(vtype=SolverVarType.BINARY, name=f"x_{t.name}_{choice_name}")
        y: dict[tuple[TransferNode, MulticastPathPlan], SolverVar] = {}
        for tr in space.transfer_nodes:
            for i, choice in enumerate(space.path_choices[tr]):
                y[(tr, choice)] = model.add_var(vtype=SolverVarType.BINARY, name=f"y_{tr.name}_choice_{i}")
        z_stop = self._create_reuse_vars()
        z_single: dict[tuple[Tensor, int], SolverVar] = {}
        for t in space.tensors_to_optimize_reuse_for:
            for stop in range(len(space.ssis[t].get_applicable_temporal_variables())):
                if space.may_single_buffer(t, stop):
                    v = model.add_var(vtype=SolverVarType.BINARY, name=f"zSingle_{t.name}_L{stop}")
                    model.add_constr(v <= z_stop[(t, stop)], name=f"zSingle_AtStop_{t.name}_L{stop}")
                    z_single[(t, stop)] = v
        slot_latency: dict[int, SolverVar] = {}
        for s in range(space.max_slot + 1):
            slot_latency[s] = model.add_var(vtype=SolverVarType.INTEGER, name=f"L_{s}")
            self.quantities.add("slot_latency", slot_latency[s]._raw, index=s)
        return DecisionVariables(x=x, y=y, z_stop=z_stop, z_single=z_single, slot_latency=slot_latency)

    def _create_reuse_vars(self) -> dict[tuple[Tensor, int], SolverVar]:
        """One binary per tensor and reuse stop, exactly one of which is set, at or beyond a declared reuse."""
        space, model = self.space, self.model
        z_stop: dict[tuple[Tensor, int], SolverVar] = {}
        optimized = set(space.tensors_to_optimize_reuse_for)
        for t in space.workload.tensors:
            levels = len(space.ssis[t].get_applicable_temporal_sizes())
            for stop in range(-1, levels):
                z_stop[(t, stop)] = model.add_var(vtype=SolverVarType.BINARY, name=f"zStop_{t.name}_L{stop}")
            name = f"zStop_Choose_One_{t.name}"
            model.add_constr(model.quicksum(z_stop[(t, s)]._raw for s in range(-1, levels)) == 1, name=name)
            self.ledger.tags[name] = ConstraintTag(rule=REUSE_CHOICE)
            if t in optimized:
                continue
            reuses = space.ssis[t].get_temporal_reuses()
            stop = next((i for i in range(len(reuses) - 1, -1, -1) if reuses[i] == Reuse.REUSE), -2)
            assert stop >= -1, f"Something went wrong for {t.name} REUSE indexing: {reuses}"
            # The declared reuse is a floor, not a target: holding a tensor across more
            # levels than asked for only removes transfers, and the capacity, routing and
            # compute-compatibility constraints already say when that does not fit.
            model.add_constr(
                model.quicksum(z_stop[(t, s)]._raw for s in range(stop, levels)) == 1,
                name=f"zStop_AtLeast_{t.name}_L{stop}",
            )
        return z_stop

    def _objective_levels(self) -> dict[str, ObjectiveLevel]:
        """The families' objective levels by name, highest priority first, those of one name summed into one."""
        levels: dict[str, ObjectiveLevel] = {}
        for family in self.families.families:
            if not isinstance(family, ObjectiveFamily):
                continue
            for level in family.objective(self.context, self.quantities):
                if (merged := levels.get(level.name)) is None:
                    levels[level.name] = level
                elif merged.priority != level.priority:
                    raise ValueError(
                        f"Objective level {level.name!r} has priority {level.priority} in {family.name!r}, "
                        f"{merged.priority} elsewhere"
                    )
                else:
                    levels[level.name] = replace(merged, expr=merged.expr + level.expr)
        if "total_latency" not in self.quantities:
            raise ValueError("The allocation needs the overlap family, which defines the latency objective")
        return dict(sorted(levels.items(), key=lambda item: -item[1].priority))

    def solve(
        self,
        *,
        time_limit_s: float,
        tee: bool = False,
        total_mac_ops: int | None = None,
        callback: Callable[[Any, int], None] | None = None,
    ) -> AllocationSolution:
        """Solve the model within ``time_limit_s`` and read the allocation back; ``tee`` prints the solver log,
        ``total_mac_ops`` of the untiled group adds its end-to-end MAC utilization to the performance report, and
        ``callback`` observes a Gurobi solve as it runs. An infeasible model raises its diagnosis."""
        self.model.set_param(SolverParams.VERBOSITY, 1 if tee else 0)
        self.model.set_param(SolverParams.TIME_LIMIT, time_limit_s)
        with span("milp_optimize"):
            self.model.optimize(callback)
        status = self.model.get_status()
        if status == "TIME_LIMIT" and self.model.get_sol_count() > 0:
            _logger.warning(
                "Allocation solve hit the %ss limit; taking the best incumbent (gap unknown)",
                time_limit_s,
            )
        elif status != "OPTIMAL":
            raise InfeasibleAllocationError(infeasibility_report(self.model, self.ledger, status))

        with span("milp_extract"):
            tensor_alloc = self.get_tensor_allocations()
            routing = self.get_transfer_routing()
            chosen_memory_cores = self.get_chosen_memory_cores(tensor_alloc)
            tensor_reuse_levels = self.get_tensor_reuse_levels()
            tensor_depths = self.get_tensor_depths()

        with span("milp_report"):
            value, q = self.model.value, self.quantities
            latency = Latency(
                total=int(value(q.get("total_latency").expr)),
                per_iteration=int(sum(slot_lat.X for slot_lat in self.vars.slot_latency.values())),
                overlap=int(value(q.get("overlap").expr)),
                fill=round(value(q.get("fill").expr)),
            )
            performance, capacity_slack = solved_reports(
                self.context, self.families, tensor_reuse_levels, latency.total, total_mac_ops
            )
            return AllocationSolution(
                tensor_placements=tensor_alloc,
                transfer_routes=routing,
                memory_cores=chosen_memory_cores,
                reuse_levels=tensor_reuse_levels,
                depths=tensor_depths,
                single_buffered=frozenset(self.get_single_buffered()),
                latency=latency,
                primary_cost=self.model.value(self.objective["latency"].expr),
                throughput_bound=self.throughput_bound(),
                solve_stats=self.model.solve_stats(),
                performance=performance,
                capacity_slack=capacity_slack,
            )

    def get_tensor_reuse_levels(self) -> TensorReuseLevels:
        reuse_levels: TensorReuseLevels = {}
        for t in self.space.workload.tensors:
            for stop in self.space.stops(t):
                if self.vars.z_stop[(t, stop)].X > self.VAR_THRESHOLD:
                    reuse_levels[t] = stop
        return reuse_levels

    def get_single_buffered(self) -> set[Tensor]:
        """The tensors whose moving window the solve holds in one buffer."""
        return {t for (t, _), z in self.vars.z_single.items() if z.X > self.VAR_THRESHOLD}

    def get_tensor_depths(self) -> TensorDepths:
        tiles_needed: TensorDepths = {}
        for t in self.space.workload.tensors:
            for stop in self.space.stops(t):
                if self.vars.z_stop[(t, stop)].X > self.VAR_THRESHOLD:
                    tiles_needed[t] = self.space.tiles_needed_levels[(t, stop)]
        return tiles_needed

    def get_transfer_routing(self) -> TransferAlloc:
        routing: TransferAlloc = {}
        for tr in self.space.transfer_nodes:
            chosen = [
                choice for choice in self.space.path_choices[tr] if self.vars.y[(tr, choice)].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{tr.name}: expected exactly one routing choice, got {chosen}")
            routing[tr] = chosen[0]
        return routing

    def get_chosen_memory_cores(self, tensor_alloc: TensorAlloc) -> MemoryAlloc:
        """The placement of the tensor each constant I/O transfer reads or writes."""
        chosen_memory_cores: MemoryAlloc = {}
        for tr in self.space.transfer_nodes:
            if not self.space.is_const_io(tr):
                continue
            tensor = self.space.constant_transfer_tensor(tr)
            if tensor in tensor_alloc:
                chosen_memory_cores[tr] = tensor_alloc[tensor]
            else:
                chosen_memory_cores[tr] = self.space.fixed_choice(tensor)
        return chosen_memory_cores

    def get_tensor_allocations(self) -> TensorAlloc:
        tensor_alloc: TensorAlloc = {}
        for t in self.space.tensor_fixed:
            tensor_alloc[t] = self.space.fixed_choice(t)
        for t in self.space.tensor_var:
            chosen = [
                choice for choice in self.space.tensor_choices[t] if self.vars.x[(t, choice)].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{t.node_name}: expected exactly one placement choice, got {chosen}")
            tensor_alloc[t] = chosen[0]
        return tensor_alloc

    def throughput_bound(self) -> float:
        """The pipelined compute bound of the steady state, from the solved allocation."""
        space, q, value = self.space, self.quantities, self.model.value
        busy: dict[Any, float] = defaultdict(float)
        for n in space.ssc_nodes:
            latencies = [space.cost_lut.get_cost(n, c).latency_total for c in space.cost_lut.get_cores(n)]
            runtime = ceil(max(latencies)) if latencies else 0
            active = float(get_active_latency(n, float(runtime), space.ssis))
            for group in space.mapping.get(n).resource_allocation:
                for core in group:
                    busy[core] += active
        per_iteration = max(busy.values(), default=0.0)
        per_iteration = max(per_iteration, float(q.get("recurrence_bound").expr))
        for shared in q.indexed("shared_busy").values() if "shared_busy" in q else ():
            per_iteration = max(per_iteration, value(shared.expr))
        chain = sum(float(v.X) for v in self.vars.slot_latency.values())
        return space.iterations * per_iteration + max(0.0, chain - per_iteration) + value(q.get("fill").expr)
