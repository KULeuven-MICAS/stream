"""Parse stage: expand every normalization (Softmax/LpNorm) into its affine sub-operators."""

from __future__ import annotations

from collections.abc import Generator

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.normalization import expand_normalizations
from stream.workload.workload import Workload


class ExpandNormalizationStage(Stage):
    reads = ("workload",)
    writes = ("workload",)

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.get("workload")
        self.ctx.set(workload=expand_normalizations(workload))

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()


_: StageCallable = ExpandNormalizationStage
