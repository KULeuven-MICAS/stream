import logging
from typing import Any

from zigzag.stages.parser.workload_parser import WorkloadParserStage as ZigZagWorkloadParserStage
from zigzag.stages.stage import Stage, StageCallable

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.onnx.model import ONNXModelParser
from stream.parser.workload_factory import WorkloadFactoryStream
from stream.workload.dnn_workload import DNNWorkloadStream

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
        self.accelerator = accelerator
        self.mapping_path = mapping_path
        self.onnx_model_parser = ONNXModelParser(workload_path, mapping_path, accelerator)

    def run(self):
        self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.get_onnx_model()
        workload = self.onnx_model_parser.get_workload()

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["mapping_path"] = self.mapping_path
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


class UserDefinedModelParserStage(ZigZagWorkloadParserStage):
    """Parses a user-provided workload from a yaml file.
    This class is very similar to WorkloadParserStage from ZigZag, the main difference being that this class creates a
    (Stream)DNNWorkload of ComputationNodes, while the ZigZag variant creates a (ZigZag) DNNWorkload of LayerNodes
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables=list_of_callables, workload=workload_path, mapping=mapping_path, **kwargs)
        self.accelerator = accelerator

    def run(self):
        workload = self.parse_workload_stream()
        self.kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload_stream(self) -> DNNWorkloadStream:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactoryStream(workload_data, mapping_data)
        return factory.create()
