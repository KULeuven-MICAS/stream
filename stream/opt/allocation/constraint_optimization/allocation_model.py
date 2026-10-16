import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeAlias

from stream.allocation.problem import AllocationProblem
from stream.allocation.solution import AllocationSolution, Latency
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.diagnosis import (
    ConstraintTag,
    StructuralRule,
    infeasibility_report,
)
from stream.opt.allocation.constraint_optimization.families import FamilySelection, ObjectiveFamily
from stream.opt.allocation.constraint_optimization.families.memory import capacity_screen
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.report import VAR_THRESHOLD, solved_reports, solver_metrics
from stream.opt.allocation.constraint_optimization.space import DecisionSpace, Placement
from stream.opt.allocation.constraint_optimization.utils import resource_key
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
TensorAlloc: TypeAlias = dict[Tensor, Placement]
TransferAlloc: TypeAlias = dict[TransferNode, MulticastPathPlan]
MemoryAlloc: TypeAlias = dict[TransferNode, Placement]


class AllocationModel:
    """The allocation model of a problem: the core decision variables -- where every movable tensor lives, which route
    each transfer takes, where each tensor's reuse stops -- the constraints and objective levels its families build on
    them, and the allocation its solve reads back."""

    def __init__(self, problem: AllocationProblem, *, families: FamilySelection, backend: str):
        self.families = families
        self.space = DecisionSpace(problem)
        self.model: SolverModel = create_solver(SolverBackend[backend], "allocation")
        self.model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
        self.quantities = QuantityRegistry()
        self.ledger = ResourceLedger()
        self._build_model()

    def _build_model(self) -> None:
        with span("capacity_screen"):
            capacity_screen(self.space, self.model)
        with span("variables"):
            self.vars = self._create_variables()
        self.context = FormulationContext(self.space, self.vars, self.model, self.quantities, self.ledger)
        for name, build in self.families.steps:
            with span(f"family_{name}"):
                build(self.context)
        with span("objective"):
            self.objective = self._objective_levels()
            self.model.set_lexicographic_objectives(list(self.objective.values()), sense="minimize")

    def _create_variables(self) -> DecisionVariables:
        space, model = self.space, self.model
        x: dict[tuple[Tensor, Placement], SolverVar] = {}
        for t in space.tensor_var:
            for choice in space.tensor_choices[t]:
                choice_name = "__".join(resource_key(c) for c in choice)
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
        """One binary per tensor and reuse stop, exactly one of which is set, at or beyond a declared reuse: the
        declared reuse is a floor, as holding a tensor longer only removes transfers where it fits."""
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
            for level in family.objective(self.context):
                if (merged := levels.get(level.name)) is None:
                    levels[level.name] = level
                elif merged.priority != level.priority:
                    raise ValueError(
                        f"Objective level {level.name!r} has priority {level.priority} in {family.name!r}, "
                        f"{merged.priority} elsewhere"
                    )
                else:
                    levels[level.name] = replace(merged, expr=merged.expr + level.expr)
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

        space, value, q = self.space, self.model.value, self.quantities
        with span("milp_extract"):
            tensor_alloc = self.get_tensor_allocations()
            routing = self.get_transfer_routing()
            memory_cores = self.get_chosen_memory_cores(tensor_alloc)
            reuse_levels = self.get_tensor_reuse_levels()
            slot_latencies = {s: float(v.X) for s, v in self.vars.slot_latency.items()}
            reuse_factors = {tr: value(factor.expr) for tr, factor in q.indexed("reuse_factor").items()}

        with span("milp_report"):
            latency = Latency(
                total=int(value(q.get("total_latency").expr)),
                per_iteration=int(sum(slot_latencies.values())),
                overlap=int(value(q.get("overlap").expr)),
                fill=round(value(q.get("fill").expr)),
            )
            performance, capacity_slack, breakdown = solved_reports(
                self.context, self.families, reuse_levels, latency.total, total_mac_ops
            )
            stats = self.model.solve_stats()
            return AllocationSolution(
                tensor_placements=tensor_alloc,
                transfer_routes=routing,
                memory_cores=memory_cores,
                reuse_levels=reuse_levels,
                depths={t: space.tiles_needed_levels[(t, stop)] for t, stop in reuse_levels.items()},
                single_buffered=frozenset(self.get_single_buffered()),
                latency=latency,
                slot_latencies=slot_latencies,
                reuse_factors=reuse_factors,
                route_cycles={tr: space.transfer_latency_for_path(tr, route) for tr, route in routing.items()},
                primary_cost=value(self.objective["latency"].expr),
                solve_stats=stats,
                metrics=solver_metrics(stats, self.model.model_size()),
                performance=performance,
                capacity_slack=capacity_slack,
                slot_latency_breakdown=breakdown,
            )

    def get_tensor_reuse_levels(self) -> TensorReuseLevels:
        reuse_levels: TensorReuseLevels = {}
        for t in self.space.workload.tensors:
            for stop in self.space.stops(t):
                if self.vars.z_stop[(t, stop)].X > VAR_THRESHOLD:
                    reuse_levels[t] = stop
        return reuse_levels

    def get_single_buffered(self) -> set[Tensor]:
        """The tensors whose moving window the solve holds in one buffer."""
        return {t for (t, _), z in self.vars.z_single.items() if z.X > VAR_THRESHOLD}

    def get_transfer_routing(self) -> TransferAlloc:
        routing: TransferAlloc = {}
        for tr in self.space.transfer_nodes:
            chosen = [choice for choice in self.space.path_choices[tr] if self.vars.y[(tr, choice)].X > VAR_THRESHOLD]
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
            chosen = [choice for choice in self.space.tensor_choices[t] if self.vars.x[(t, choice)].X > VAR_THRESHOLD]
            if len(chosen) != 1:
                raise ValueError(f"{t.node_name}: expected exactly one placement choice, got {chosen}")
            tensor_alloc[t] = chosen[0]
        return tensor_alloc
