"""Annotate the workload with AccessRelation-aware fusion edges.

A passthrough stage: it classifies every compute-to-compute edge with
:func:`~stream.workload.fusion.analysis.workload_fusion_edges` and records the result under
``fusion_edges`` in the context. Data-dependent reads (gather / MoE dispatch-combine) are hard fusion
barriers; a normalization's reduced axis is a per-axis barrier. On its own this stage changes no
allocation behaviour -- it exposes the fusion structure for a downstream cut-point provider or a
visualization -- so it is safe to insert anywhere after the workload exists.
"""

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
