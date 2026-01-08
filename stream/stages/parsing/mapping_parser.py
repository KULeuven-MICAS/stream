import logging

from stream.parser.mapping_parser import MappingParser
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class MappingParserStage(Stage):
    REQUIRED_FIELDS = ("accelerator", "workload", "mapping_path")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        mapping_path = self.ctx.require_value("mapping_path", self.__class__.__name__)
        self.mapping_parser = MappingParser(mapping_path, self.workload, self.accelerator)

    def run(self):
        mapping = self.mapping_parser.run()

        self.ctx.set(
            mapping=mapping,
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
