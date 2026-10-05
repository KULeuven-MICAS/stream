import os

from stream.allocation.allocation import Allocation, solved_iteration_spaces, solved_mapping
from stream.allocation.artifacts import SolveProgress, write_artifacts, write_infeasible_model
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.profiling import span
from stream.stages.stage import Stage


class AllocationStage(Stage):
    """Solve the allocation problem -- where each tensor lives and which route each transfer takes -- and hand
    downstream the allocation it yields."""

    reads = ("allocation_problem", "output_path", "backend", "families", "time_limit_s", "solver_log", "artifacts")
    optional_reads = ("total_mac_ops",)
    writes = ("allocation", "workload", "mapping")

    def run(self):
        ctx = self.ctx
        problem, backend, families = ctx.get("allocation_problem"), ctx.get("backend"), ctx.get("families")
        output_path = os.path.join(ctx.get("output_path"), "allocation")
        with span("milp_build"):
            model = AllocationModel(problem, families=families, backend=backend)
        progress = SolveProgress() if ctx.get("artifacts") else None
        try:
            solution = model.solve(
                tee=ctx.get("solver_log"),
                time_limit_s=ctx.get("time_limit_s"),
                total_mac_ops=ctx.get("total_mac_ops"),
                callback=progress,
            )
        except InfeasibleAllocationError:
            write_infeasible_model(model.model, output_path)
            raise
        with span("apply_solution"):
            allocation = Allocation(
                problem=problem,
                mapping=solved_mapping(problem.workload, problem.mapping, solution),
                ssis=solved_iteration_spaces(problem.workload, problem.ssis, solution.reuse_levels),
                backend=backend,
                families=families.specs(),
                solution=solution,
            )
        if progress is not None:
            write_artifacts(output_path, allocation, progress)
        ctx.set(allocation=allocation, workload=problem.workload, mapping=allocation.mapping)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], ctx)
        yield from sub_stage.run()

    def is_leaf(self) -> bool:
        return False
