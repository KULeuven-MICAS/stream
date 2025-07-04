from typing import Any

from zigzag.stages.parser.workload_parser import WorkloadParserStage as ZigZagWorkloadParserStage
from zigzag.stages.stage import StageCallable

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.mapping_parser import MappingParser
from stream.parser.workload_factory import WorkloadFactoryStream
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import InterCoreMappingAttributes


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
        self.mapping_parser = MappingParser(mapping_path)

    def run(self):
        all_mappings = self.mapping_parser.run()
        workload = self.parse_workload_stream(all_mappings)
        self.kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        yield from sub_stage.run()

    def parse_workload_stream(self, all_mappings: dict[str, InterCoreMappingAttributes]) -> DNNWorkloadStream:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactoryStream(workload_data, mapping_data)
        return factory.create(all_mappings)
