"""Annotate the workload with structural block classes (repeated-subgraph detection).

A passthrough stage: it computes computation-node equivalence classes (see
:func:`~stream.workload.structure.block_detect.find_repeated_blocks`) and records a
``block_classes`` map ``{node_name: class_id}`` in the context. Downstream allocation/fusion may
reuse a class representative's decisions across its occurrences; on its own this stage changes no
behaviour, so it is safe to insert early in any pipeline.
"""

from __future__ import annotations

from collections.abc import Generator

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.structure.block_detect import find_repeated_blocks
from stream.workload.workload import Workload


class StructuralDedupStage(Stage):
    REQUIRED_FIELDS = ("workload",)

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.require_value("workload", self.__class__.__name__)
        block_class_id: dict[str, int] = {}
        for class_id, block in enumerate(find_repeated_blocks(workload)):
            for node in block.nodes:
                block_class_id[node.name] = class_id
        self.ctx.set(block_classes=block_class_id)

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()


_: StageCallable = StructuralDedupStage
