"""Cost-stage runs on the 2-conv workload on four TPU-like cores, for the core-cost backend tests."""

from __future__ import annotations

from pathlib import Path

from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import open_yaml

from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage
from stream.stages.stage import Stage

REPO = Path(__file__).parents[2]
ACCELERATOR = REPO / "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
WORKLOAD = REPO / "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
MAPPING = REPO / "stream/inputs/testing/mapping/2conv_1_32_32_8_16_32_3.yaml"


class _Leaf(Stage):
    """The stage after the cost stage, which these runs never reach."""

    def __init__(self) -> None:
        super().__init__([])

    def is_leaf(self) -> bool:
        return True

    def run(self):
        return iter(())


def inputs():
    """The workload, the accelerator and the mapping of the run."""
    validator = AcceleratorValidator(open_yaml(str(ACCELERATOR)), str(ACCELERATOR))
    assert validator.validate()
    accelerator = AcceleratorFactory(validator.normalized_data).create()
    parser = ONNXModelParser(str(WORKLOAD))
    parser.run()
    return parser.workload, accelerator, MappingParser(str(MAPPING), parser.workload, accelerator).run()


def priced(output: Path, **options) -> CoreCostEstimationStage:
    """The cost stage run with ``options``, its cost table cached in ``output``."""
    workload, accelerator, mapping = inputs()
    stage = CoreCostEstimationStage(
        [_Leaf],
        StageContext.from_kwargs(
            workload=workload,
            accelerator=accelerator,
            mapping=mapping,
            loma_lpf_limit=6,
            temporal_mapping_type=TemporalMappingType.UNEVEN,
            output_path=str(output),
            **options,
        ),
    )
    stage.update_cost_lut()
    stage.cost_lut.save()
    return stage
