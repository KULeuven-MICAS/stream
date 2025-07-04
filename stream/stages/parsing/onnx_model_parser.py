import logging
from typing import Any

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload_path = workload_path
        self.accelerator = accelerator
        self.mapping_parser = MappingParser(mapping_path)

    def run(self):
        all_mappings = self.mapping_parser.run()
        onnx_model_parser = ONNXModelParser(self.workload_path, all_mappings, self.accelerator)
        onnx_model_parser.run()
        onnx_model = onnx_model_parser.onnx_model
        workload = onnx_model_parser.workload

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["all_mappings"] = all_mappings
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        yield from sub_stage.run()
