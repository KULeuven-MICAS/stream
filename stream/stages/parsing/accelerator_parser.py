import logging
from typing import Any

from zigzag.stages.stage import Stage, StageCallable
from zigzag.utils import open_yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator

logger = logging.getLogger(__name__)


class AcceleratorParserStage(Stage):
    """Parse to parse an accelerator from a user-defined yaml file."""

    def __init__(self, list_of_callables: list[StageCallable], *, accelerator: str, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        assert accelerator.split(".")[-1] == "yaml", "Expected a yaml file as accelerator input"
        self.accelerator_yaml_path = accelerator

    def run(self):
        accelerator = self.parse_accelerator()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_accelerator(self) -> Accelerator:
        accelerator_data = open_yaml(self.accelerator_yaml_path)

        validator = AcceleratorValidator(accelerator_data, self.accelerator_yaml_path)
        accelerator_data = validator.normalized_data
        validate_success = validator.validate()
        if not validate_success:
            raise ValueError("Failed to validate user provided accelerator.")

        factory = AcceleratorFactory(accelerator_data)
        return factory.create()
