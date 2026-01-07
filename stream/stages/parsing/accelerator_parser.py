import logging

from zigzag.utils import open_yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class AcceleratorParserStage(Stage):
    """Parse to parse an accelerator from a user-defined yaml file."""

    REQUIRED_FIELDS = ("accelerator",)

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)

    def run(self):
        if isinstance(self.accelerator, Accelerator):
            accelerator = self.accelerator
        else:
            assert self.accelerator.split(".")[-1] == "yaml", "Expected a yaml file as accelerator input"
            accelerator = self.parse_accelerator_from_yaml(self.accelerator)

        self.ctx.set(accelerator=accelerator)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def parse_accelerator_from_yaml(self, yaml_path: str) -> Accelerator:
        accelerator_data = open_yaml(yaml_path)

        validator = AcceleratorValidator(accelerator_data, yaml_path)
        accelerator_data = validator.normalized_data
        validate_success = validator.validate()
        if not validate_success:
            raise ValueError("Failed to validate user provided accelerator.")

        factory = AcceleratorFactory(accelerator_data)
        return factory.create()
