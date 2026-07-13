"""Opt-in parse stage: expand every normalization (Softmax/LpNorm) into its affine sub-operators.

Off by default (the monolithic ``NormalizationNode`` is what the current pipeline and the AIE codegen
schedule). When ``expand_normalizations`` is set in the context, the workload the downstream cost and
fusion stages see carries explicit ReduceMax/Exp/ReduceSum/Div sub-ops, so a safe softmax's two
reduction passes are counted (not under-counted as one element-wise pass) and its reduction shows up as
an ordinary affine reduction. The sub-ops stay tagged with their origin kernel so a codegen backend can
re-collapse them into one native softmax later (see ``collapse_fused_kernels``).
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
        if self.ctx.get("expand_normalizations", False):
            workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
            self.ctx.set(workload=expand_normalizations(workload))

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()


_: StageCallable = ExpandNormalizationStage
