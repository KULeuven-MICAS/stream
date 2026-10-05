import logging
import os

from stream.allocation.allocation import Allocation, solved_iteration_spaces, solved_mapping
from stream.allocation.artifacts import SolveProgress, write_artifacts, write_infeasible_model
from stream.allocation.problem import AllocationProblem
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.opt.allocation.constraint_optimization.families import FamilySelection
from stream.profiling import span
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class AllocationStage(Stage):
    """Solve the allocation problem -- where each tensor lives and which route each transfer takes -- and hand
    downstream the allocation it yields."""

    reads = ("allocation_problem", "output_path", "backend", "families", "time_limit_s", "solver_log", "artifacts")
    optional_reads = ("total_mac_ops",)
    writes = ("allocation", "workload", "mapping")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.problem: AllocationProblem = self.ctx.get("allocation_problem")
        self.output_path = os.path.join(self.ctx.get("output_path"), "allocation")
        self.backend: str = self.ctx.get("backend")
        self.families: FamilySelection = self.ctx.get("families")
        self.time_limit_s: float = self.ctx.get("time_limit_s")
        self.solver_log: bool = self.ctx.get("solver_log")
        self.artifacts: bool = self.ctx.get("artifacts")

    def run(self):
        problem = self.problem
        with span("milp_build"):
            model = AllocationModel(problem, families=self.families, backend=self.backend)
        progress = SolveProgress() if self.artifacts else None
        try:
            solution = model.solve(
                tee=self.solver_log,
                time_limit_s=self.time_limit_s,
                total_mac_ops=self.ctx.get("total_mac_ops"),
                callback=progress,
            )
        except InfeasibleAllocationError:
            write_infeasible_model(model.model, self.output_path)
            raise
        with span("apply_solution"):
            allocation = Allocation(
                problem=problem,
                mapping=solved_mapping(problem.workload, problem.mapping, solution),
                ssis=solved_iteration_spaces(problem.workload, problem.ssis, solution.reuse_levels),
                backend=self.backend,
                families=self.families.specs(),
                solution=solution,
            )
        if progress is not None:
            write_artifacts(self.output_path, allocation, progress)
        self.ctx.set(allocation=allocation, workload=problem.workload, mapping=allocation.mapping)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def is_leaf(self) -> bool:
        return False
