from pathlib import Path

from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import open_yaml

from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.stages.estimation.zigzag_core_mapping_estimation import CoreCostEstimationStage


def build_accelerator(hardware_path: str):
    data = open_yaml(hardware_path)
    validator = AcceleratorValidator(data, hardware_path)
    data = validator.normalized_data
    assert validator.validate(), "Accelerator validation failed."
    factory = AcceleratorFactory(data)
    return factory.create()


def build_workload(workload_path: str, mapping_path: str, accelerator):
    mappings = MappingParser(mapping_path).run()
    parser = ONNXModelParser(workload_path, mappings, accelerator)
    parser.run()
    return parser.workload


def generate_cost_lut(tmp_path: Path) -> Path:
    hardware = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    workload_path = "stream/inputs/testing/workload/2_conv.onnx"
    mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"

    accelerator = build_accelerator(hardware)
    workload = build_workload(workload_path, mapping_path, accelerator)

    cost_lut_path = tmp_path / "outputs" / "test-2conv" / "cost_lut.pickle"
    cost_lut_path.parent.mkdir(parents=True, exist_ok=True)

    estimator = CoreCostEstimationStage(
        [_NoOpLeafStage],
        workload=workload,
        accelerator=accelerator,
        loma_lpf_limit=6,
        cost_lut_path=str(cost_lut_path),
        layer_stacks=[(0, 1)],
        temporal_mapping_type=TemporalMappingType.UNEVEN,
    )
    estimator.update_cost_lut()
    estimator.cost_lut.save()
    return cost_lut_path


def test_core_cost_lut_caches_and_loads(tmp_path):
    cost_lut_path = generate_cost_lut(tmp_path)
    assert cost_lut_path.exists()

    lut_first = CoreCostLUT(str(cost_lut_path))
    assert lut_first.get_nodes(), "LUT should contain at least one node after first run."

    # Re-run with existing cache to ensure it loads successfully
    estimator = CoreCostEstimationStage(
        [_NoOpLeafStage],
        workload=build_workload(
            "stream/inputs/testing/workload/2_conv.onnx",
            "stream/inputs/examples/mapping/tpu_like_quad_core.yaml",
            build_accelerator("stream/inputs/examples/hardware/tpu_like_quad_core.yaml"),
        ),
        accelerator=build_accelerator("stream/inputs/examples/hardware/tpu_like_quad_core.yaml"),
        loma_lpf_limit=6,
        cost_lut_path=str(cost_lut_path),
        layer_stacks=[(0, 1)],
        temporal_mapping_type=TemporalMappingType.UNEVEN,
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
