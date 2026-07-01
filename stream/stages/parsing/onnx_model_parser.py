import logging
import os

from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)

_VIZ_NODE_LIMIT = 30


class ONNXModelParserStage(Stage):
    REQUIRED_FIELDS = ("workload_path", "output_path")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.workload_path = self.ctx.get("workload_path")
        self.output_path = self.ctx.get("output_path")
        self.workload_visualization_path = os.path.join(self.output_path, "workload_graph.png")

    def run(self):
        onnx_model_parser = ONNXModelParser(self.workload_path)
        onnx_model_parser.run()
        onnx_model = onnx_model_parser.onnx_model
        workload = onnx_model_parser.workload
        node_count = len(list(workload.nodes))
        if node_count > _VIZ_NODE_LIMIT:
            logger.warning("Skipping workload visualization: %d nodes exceeds limit %d", node_count, _VIZ_NODE_LIMIT)
        else:
            # The workload graph PNG is a debug-only artifact rendered via graphviz
            # `dot` (pydot). Don't let a missing/broken graphviz abort code
            # generation -- e.g. environments without graphviz installed.
            try:
                workload.visualize(self.workload_visualization_path)
            except Exception as e:  # noqa: BLE001 -- visualization is best-effort
                logger.warning("Skipping workload visualization (%s): %s", type(e).__name__, e)

        self.ctx.set(
            onnx_model=onnx_model,
            workload=workload,
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
