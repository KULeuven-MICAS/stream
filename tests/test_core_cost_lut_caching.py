from pathlib import Path

from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import open_yaml

from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage

ACCELERATOR_PATH = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
WORKLOAD_PATH = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
MAPPING_PATH = "stream/inputs/testing/mapping/2conv_1_32_32_8_16_32_3.yaml"


def build_accelerator(hardware_path: str):
    data = open_yaml(hardware_path)
    validator = AcceleratorValidator(data, hardware_path)
    data = validator.normalized_data
    assert validator.validate(), "Accelerator validation failed."
    factory = AcceleratorFactory(data)
    return factory.create()


def build_workload(workload_path: str):
    parser = ONNXModelParser(workload_path)
    parser.run()
    workload = parser.workload
    return workload


def build_mapping(workload, accelerator, mapping_path):
    return MappingParser(mapping_path, workload, accelerator).run()


def generate_cost_lut(cost_lut_path: Path) -> None:
    output_path = cost_lut_path.parent

    accelerator = build_accelerator(ACCELERATOR_PATH)
    workload = build_workload(WORKLOAD_PATH)
    mapping = build_mapping(workload, accelerator, MAPPING_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    estimator = CoreCostEstimationStage(
        [_NoOpLeafStage],
        StageContext.from_kwargs(
            workload=workload,
            accelerator=accelerator,
            mapping=mapping,
            loma_lpf_limit=6,
            cost_lut_path=str(cost_lut_path),
            temporal_mapping_type=TemporalMappingType.UNEVEN,
            output_path=output_path,
        ),
    )
    estimator.update_cost_lut()
    estimator.cost_lut.save()


def test_core_cost_lut_caches_and_loads(tmp_path):
    cost_lut_path = tmp_path / "core_cost_lut.pickle"
    generate_cost_lut(cost_lut_path)
    assert cost_lut_path.exists()

    lut_first = CoreCostLUT(str(cost_lut_path))
    assert lut_first.get_nodes(), "LUT should contain at least one node after first run."

    workload = build_workload(WORKLOAD_PATH)
    accelerator = build_accelerator(ACCELERATOR_PATH)
    mapping = build_mapping(workload, accelerator, MAPPING_PATH)

    # Re-run with existing cache to ensure it loads successfully
    estimator = CoreCostEstimationStage(
        [_NoOpLeafStage],
        StageContext.from_kwargs(
            workload=workload,
            accelerator=accelerator,
            mapping=mapping,
            loma_lpf_limit=6,
            cost_lut_path=str(cost_lut_path),
            temporal_mapping_type=TemporalMappingType.UNEVEN,
            output_path=tmp_path,
            nb_spatial_mappings_generated=1,
        ),
    )
    estimator.update_cost_lut()
    estimator.cost_lut.save()

    lut_second = CoreCostLUT(str(cost_lut_path))
    assert lut_second.get_nodes(), "LUT should load cached entries on subsequent runs."


class _NoOpLeafStage(CoreCostEstimationStage.__bases__[0]):  # type: ignore[misc]
    """Simple leaf stage that yields nothing."""

    def __init__(self):
        super().__init__([])

    def is_leaf(self):
        return True

    def run(self):
        return iter(())
