from stream.allocation.lowering import lower_steady_state
from stream.stages.stage import Stage


class SteadyStateLoweringStage(Stage):
    """Lower the fused group to the allocation problem of its steady state."""

    reads = ("workload", "accelerator", "mapping", "cost_lut", "fusion_splits", "nb_cols_to_use")
    writes = ("allocation_problem",)

    def run(self):
        problem = lower_steady_state(
            self.ctx.get("workload"),
            self.ctx.get("accelerator"),
            self.ctx.get("mapping"),
            self.ctx.get("fusion_splits"),
            self.ctx.get("cost_lut"),
            self.ctx.get("nb_cols_to_use"),
        )
        self.ctx.set(allocation_problem=problem)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def is_leaf(self) -> bool:
        return False
