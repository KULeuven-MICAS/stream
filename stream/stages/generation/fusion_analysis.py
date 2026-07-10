"""Passthrough stage: record ``workload_fusion_edges(workload)`` under ``fusion_edges`` in the context."""

from __future__ import annotations

from collections.abc import Generator

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.fusion.analysis import workload_fusion_edges
from stream.workload.workload import Workload


class FusionAnalysisStage(Stage):
    REQUIRED_FIELDS = ("workload",)

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.ctx.set(fusion_edges=workload_fusion_edges(workload))

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()


_: StageCallable = FusionAnalysisStage
