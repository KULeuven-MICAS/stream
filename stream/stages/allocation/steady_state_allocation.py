import logging
import os

from stream.allocation.artifacts import write_artifacts
from stream.allocation.problem import SteadyStateProblem
from stream.allocation.schedule import SteadyStateSchedule, solved_iteration_spaces, solved_mapping
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.opt.solver import ConstraintSelection
from stream.profiling import span
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)

DEFAULT_TIME_LIMIT_S = 300


class AllocationStage(Stage):
    """Solve the allocation of the steady-state problem -- where each tensor lives and which route each
    transfer takes -- and hand downstream the schedule it yields.

    Reads: steady_state_problem, output_path, backend, constraint_selection, total_mac_ops, time_limit_s, solver_log
    Writes: allocation, workload, mapping
    """

    REQUIRED_FIELDS = ("steady_state_problem", "output_path")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.problem: SteadyStateProblem = self.ctx.get("steady_state_problem")
        self.output_path = os.path.join(self.ctx.get("output_path"), "tetra")
        self.backend: str = self.ctx.get("backend", "ORTOOLS_GSCIP")
        self.constraint_selection = self.ctx.get("constraint_selection") or ConstraintSelection()
        self.time_limit_s: float = self.ctx.get("time_limit_s", DEFAULT_TIME_LIMIT_S)
        self.solver_log: bool = self.ctx.get("solver_log", False)

    def run(self):
        problem = self.problem
        with span("milp_build"):
            allocator = TransferAndTensorAllocator(
                problem.workload,
                problem.timeslots,
                accelerator=problem.accelerator,
                iterations=problem.iterations,
                ssis=problem.ssis,
                multiplicities=problem.multiplicities,
                mapping=problem.mapping,
                cost_lut=problem.cost_lut,
                nb_cols_to_use=problem.nb_cols_to_use,
                context=problem.transfer_context,
                output_path=self.output_path,
                backend=self.backend,
                constraint_selection=self.constraint_selection,
            )
        solution = allocator.solve(
            tee=self.solver_log, time_limit_s=self.time_limit_s, total_mac_ops=self.ctx.get("total_mac_ops")
        )
        with span("apply_solution"):
            schedule = SteadyStateSchedule(
                source_workload=problem.source_workload,
                workload=problem.workload,
                mapping=solved_mapping(problem.workload, problem.mapping, solution),
                ssis=solved_iteration_spaces(problem.workload, problem.ssis, solution.reuse_levels),
                iterations=problem.iterations,
                fusion_splits=problem.fusion_splits,
                accelerator=problem.accelerator,
                cost_lut=problem.cost_lut,
                backend=self.backend,
                constraint_selection=self.constraint_selection,
                solution=solution,
            )
        write_artifacts(allocator, schedule)
        self.ctx.set(allocation=schedule, workload=schedule.workload, mapping=schedule.mapping)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def is_leaf(self) -> bool:
        return False
