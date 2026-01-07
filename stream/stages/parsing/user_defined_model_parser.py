from zigzag.stages.parser.workload_parser import WorkloadParserStage as ZigZagWorkloadParserStage

from stream.parser.mapping_parser import MappingParser
from stream.parser.workload_factory import WorkloadFactoryStream
from stream.stages.context import StageContext
from stream.stages.stage import StageCallable
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
        ctx: StageContext,
    ):
        super().__init__(
            list_of_callables=list_of_callables,
            workload=ctx.require_value("workload_path", self.__class__.__name__),
            mapping=ctx.require_value("mapping_path", self.__class__.__name__),
        )
        self.ctx = ctx
        self.accelerator = ctx.require_value("accelerator", self.__class__.__name__)
        self.mapping_parser = MappingParser(ctx.require_value("mapping_path", self.__class__.__name__))

    def run(self):
        all_mappings = self.mapping_parser.run()
        workload = self.parse_workload_stream(all_mappings)
        self.ctx.set(accelerator=self.accelerator, workload=workload, all_mappings=all_mappings)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def parse_workload_stream(self, all_mappings: dict[str, InterCoreMappingAttributes]) -> DNNWorkloadStream:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactoryStream(workload_data, mapping_data)
        return factory.create(all_mappings)
