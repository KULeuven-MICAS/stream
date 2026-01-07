import logging

from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    REQUIRED_FIELDS = ("workload_path", "mapping_path", "accelerator")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.workload_path = self.ctx.require_value("workload_path", self.__class__.__name__)
        mapping_path = self.ctx.require_value("mapping_path", self.__class__.__name__)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.mapping_parser = MappingParser(mapping_path)

    def run(self):
        all_mappings = self.mapping_parser.run()
        onnx_model_parser = ONNXModelParser(self.workload_path, all_mappings, self.accelerator)
        onnx_model_parser.run()
        onnx_model = onnx_model_parser.onnx_model
        workload = onnx_model_parser.workload

        self.ctx.set(
            accelerator=self.accelerator,
            all_mappings=all_mappings,
            onnx_model=onnx_model,
            workload=workload,
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
