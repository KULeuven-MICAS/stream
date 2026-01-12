import logging
import os

from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


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
        workload.visualize_to_file(self.workload_visualization_path)

        self.ctx.set(
            onnx_model=onnx_model,
            workload=workload,
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
