import logging

from stream.allocation.lowering import lower_steady_state
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class SteadyStateLoweringStage(Stage):
    """Lower the fused group to the steady-state problem its allocation solves."""

    reads = ("workload", "accelerator", "mapping", "cost_lut", "fusion_splits")
    optional_reads = ("nb_cols_to_use",)
    writes = ("steady_state_problem",)

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.nb_cols_to_use: int = self.ctx.get("nb_cols_to_use", 4)

    def run(self):
        problem = lower_steady_state(
            self.ctx.get("workload"),
            self.ctx.get("accelerator"),
            self.ctx.get("mapping"),
            self.ctx.get("fusion_splits"),
            self.ctx.get("cost_lut"),
            self.nb_cols_to_use,
        )
        self.ctx.set(steady_state_problem=problem)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def is_leaf(self) -> bool:
        return False
