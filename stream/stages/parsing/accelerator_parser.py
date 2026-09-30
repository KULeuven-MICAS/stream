import logging

from zigzag.utils import open_yaml

from stream.compiler.kernels.library import KernelLibrary
from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


def parse_accelerator(yaml_path: str) -> Accelerator:
    """The accelerator a hardware YAML describes, validated."""
    assert yaml_path.rsplit(".", maxsplit=1)[-1] == "yaml", "Expected a yaml file as accelerator input"
    validator = AcceleratorValidator(open_yaml(yaml_path), yaml_path)
    accelerator_data = validator.normalized_data
    if not validator.validate():
        raise ValueError("Failed to validate user provided accelerator.")
    return AcceleratorFactory(accelerator_data).create()


class AcceleratorParserStage(Stage):
    """Parse to parse an accelerator from a user-defined yaml file."""

    REQUIRED_FIELDS = ("accelerator",)

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)

    def run(self):
        accelerator = (
            self.accelerator if isinstance(self.accelerator, Accelerator) else parse_accelerator(self.accelerator)
        )
        if (library := self.ctx.get("kernel_library")) is not None:
            accelerator.kernel_library = KernelLibrary.load(library)

        self.ctx.set(accelerator=accelerator)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
