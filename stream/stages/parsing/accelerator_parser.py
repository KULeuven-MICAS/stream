import logging
import os

from zigzag.utils import open_yaml

from stream.compiler.kernels.library import KernelLibrary
from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.visualization.hardware_graph import draw_hardware

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

    reads = ("accelerator",)
    optional_reads = ("kernel_library", "output_path")
    writes = ("accelerator",)

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.get("accelerator")

    def run(self):
        accelerator = (
            self.accelerator if isinstance(self.accelerator, Accelerator) else parse_accelerator(self.accelerator)
        )
        if (library := self.ctx.get("kernel_library")) is not None:
            accelerator.kernel_library = KernelLibrary.load(library)
        if (output_path := self.ctx.get("output_path")) is not None:
            # A picture for reading the hardware as Stream understands it; a failure to draw it must not stop a solve.
            try:
                os.makedirs(output_path, exist_ok=True)
                draw_hardware(accelerator, os.path.join(output_path, "hardware_graph.svg"))
            except Exception as e:  # noqa: BLE001 -- visualization is best-effort
                logger.warning("Skipping hardware visualization (%s): %s", type(e).__name__, e)

        self.ctx.set(accelerator=accelerator)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
