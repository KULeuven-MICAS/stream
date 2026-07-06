"""Annotate the workload with auto-proposed fusion regions (plan Phase 3).

A passthrough stage: it runs the greedy, capacity-bounded fusion proposer
(:func:`~stream.workload.fusion.proposer.propose_fusion_regions`) and records the result under
``proposed_fusion_regions`` in the context. It proposes regions from the affine analysis alone and
changes no allocation behaviour on its own, so it is safe to insert anywhere after the workload exists.
The near-memory capacity is read from ``fusion_capacity_elements`` (default: unbounded, i.e. every
legal fusion is proposed as one region).
"""

from __future__ import annotations

from collections.abc import Generator

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.fusion.proposer import propose_fusion_regions
from stream.workload.workload import Workload


class FusionProposalStage(Stage):
    REQUIRED_FIELDS = ("workload",)

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
        capacity = self.ctx.get("fusion_capacity_elements")
        if capacity is None:
            capacity = 2**63  # unbounded: propose every legal fusion as a single region
        self.ctx.set(proposed_fusion_regions=propose_fusion_regions(workload, int(capacity)))

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()


_: StageCallable = FusionProposalStage
