"""Parse stage: expand every normalization (Softmax/LpNorm) into its affine sub-operators.

Runs right after parsing in the generic CO pipeline, so the workload the cost, fusion and MILP stages
see carries explicit ReduceMax/Exp/ReduceSum/Div sub-ops. A safe softmax's two reduction passes are then
counted (not under-counted as one element-wise pass), and its reduction is an ordinary affine reduction
the fusion/tiling analysis reads directly. The sub-ops stay tagged with their origin kernel, so a codegen
backend can re-collapse them into one native softmax later (``collapse_fused_kernels``). Dispatch goes
through the shared ``stream.workload.decompose`` registry.
"""

from __future__ import annotations

from collections.abc import Generator

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.normalization import expand_normalizations
from stream.workload.workload import Workload


class ExpandNormalizationStage(Stage):
    REQUIRED_FIELDS = ("workload",)

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.ctx.set(workload=expand_normalizations(workload))

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()


_: StageCallable = ExpandNormalizationStage
