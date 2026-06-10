"""Tests for GenericMappingGenerator and the generic CO pipeline (MAP-01 through MAP-04, FMT-05).

The two-conv workload is used as a fast single-group proxy (< 60s).
The Conv-Relu-Flatten-Gemm workload tests multi-group pipeline mechanics (RES-01/RES-02).
"""

import tempfile

import pytest
import yaml

from stream.api import optimize_allocation_co_generic
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_conv_relu_flatten_gemm import (
    make_conv_relu_flatten_gemm_workload,
)
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

_ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
_WORKLOAD_CONFIG = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)


def _parse_workload_and_accelerator(workload_path: str | None = None):
    """Helper: parse hardware + workload into objects via the pipeline stages."""
    if workload_path is None:
        workload_path = make_2_conv_workload(_WORKLOAD_CONFIG)
    ctx = StageContext.from_kwargs(
        accelerator=_ACCELERATOR,
        workload_path=workload_path,
        output_path=tempfile.mkdtemp(),
    )
    stages = [AcceleratorParserStage, ONNXModelParserStage, LeafStage]
    mainstage = MainStage(stages, ctx)
    ctxs = mainstage.run()
    assert len(ctxs) == 1
    return ctxs[0].get("accelerator"), ctxs[0].get("workload")


def test_generator_produces_all_fields():
    """MAP-01: GenericMappingGenerator produces mapping with core_allocation,
    inter_core_tiling, fused_groups, and intra_core_tiling populated."""
    accelerator, workload = _parse_workload_and_accelerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, sub_workloads = gen.generate_all_groups()

        assert len(paths) >= 1, "Expected at least one group mapping YAML"

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert "layers" in data, "Missing 'layers' key"
            assert len(data["layers"]) > 0, "Empty layers list"
            assert "fused_groups" in data, "Missing 'fused_groups' key"

            for layer in data["layers"]:
                assert "name" in layer, "Layer missing 'name'"
                assert "core_allocation" in layer, f"Layer {layer['name']} missing 'core_allocation'"
                assert isinstance(layer["core_allocation"], list), "core_allocation not a list"
                assert all(isinstance(slot, list) for slot in layer["core_allocation"]), (
                    "core_allocation entries must be lists (nested format)"
                )


def test_single_fused_group():
    """MAP-02: Generated mapping contains exactly one FusedGroup with valid intra_core_tiling."""
    accelerator, workload = _parse_workload_and_accelerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, sub_workloads = gen.generate_all_groups()

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert len(data["fused_groups"]) == 1, f"Expected 1 fused group, got {len(data['fused_groups'])}"
            fg = data["fused_groups"][0]
            assert "layers" in fg and len(fg["layers"]) > 0, "FusedGroup has empty layers"
            assert "intra_core_tiling" in fg and len(fg["intra_core_tiling"]) > 0, (
                "FusedGroup has empty intra_core_tiling"
            )


def test_mapping_validates():
    """MAP-03: Generated mapping passes MappingValidator validation."""
    accelerator, workload = _parse_workload_and_accelerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, sub_workloads = gen.generate_all_groups()

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            validator = MappingValidator(data)
            is_valid = validator.validate()
            assert is_valid, f"MappingValidator errors: {validator.errors}"


def test_pipeline_end_to_end():
    """MAP-04: optimize_allocation_co_generic completes end-to-end for two-conv TPU workload.

    Two-conv is used as a fast single-group proxy (< 60s) for quick iteration.
    test_pipeline_multi_group covers multi-group pipeline mechanics (RES-01/RES-02).
    """
    workload_path = make_2_conv_workload(_WORKLOAD_CONFIG)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=workload_path,
            experiment_id="test-generic-two-conv",
            output_path=tmpdir,
            skip_if_exists=False,
        )

    scheduler: SteadyStateScheduler = ctx.get("scheduler")
    assert scheduler is not None, "No scheduler in context"
    assert scheduler.latency_total > 0, f"Expected positive latency, got {scheduler.latency_total}"
    assert scheduler.iterations > 0, f"Expected positive iterations, got {scheduler.iterations}"

    # Verify total_latency was set by FusionGroupIterationStage
    total_latency = ctx.get("total_latency")
    assert total_latency is not None, "total_latency not set in context"
    assert total_latency > 0, f"Expected positive total_latency, got {total_latency}"

    # Verify per-group latencies are tracked
    group_latencies = ctx.get("group_latencies")
    assert group_latencies is not None, "group_latencies not set in context"
    assert len(group_latencies) >= 1, f"Expected at least 1 group, got {len(group_latencies)}"
    assert all(lat > 0 for lat in group_latencies.values()), "All group latencies must be positive"


@pytest.mark.slow  # multi-group MILP run; too slow / near-timeout on CI runners
@pytest.mark.timeout(120)
def test_pipeline_multi_group():
    """RES-01/RES-02: optimize_allocation_co_generic completes for a multi-group workload.

    Uses a synthetic Conv->Relu->Flatten->Gemm workload that splits into 2 fusion groups
    via FusionEdge parsing. Tests the same multi-group pipeline mechanics as ResNet18
    (FusionEdge splitting, per-group iteration, latency aggregation) in seconds.
    """
    workload_path = make_conv_relu_flatten_gemm_workload()

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=workload_path,
            experiment_id="test-multi-group",
            output_path=tmpdir,
            skip_if_exists=False,
        )

    # Verify scheduler result
    scheduler: SteadyStateScheduler = ctx.get("scheduler")
    assert scheduler is not None, "No scheduler in context"
    assert scheduler.latency_total > 0, f"Expected positive latency, got {scheduler.latency_total}"

    # Verify total_latency aggregated across all groups
    total_latency = ctx.get("total_latency")
    assert total_latency is not None, "total_latency not set in context"
    assert total_latency > 0, f"Expected positive total_latency, got {total_latency}"

    # Verify per-group latencies (RES-01: positive latency; RES-02: multi-group handling)
    group_latencies = ctx.get("group_latencies")
    assert group_latencies is not None, "group_latencies not set in context"
    assert len(group_latencies) == 2, f"Expected 2 groups (Conv+Relu and Gemm), got {len(group_latencies)}"
    assert all(lat > 0 for lat in group_latencies.values()), (
        f"All group latencies must be positive, got {group_latencies}"
    )
    # Verify aggregation: total = sum of per-group
    assert abs(total_latency - sum(group_latencies.values())) < 1e-6, (
        f"total_latency ({total_latency}) != sum of group_latencies ({sum(group_latencies.values())})"
    )


def test_tpu_yaml_validates():
    """FMT-05: Existing TPU mapping YAML validates against MappingValidator schema."""
    tpu_mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    with open(tpu_mapping_path) as f:
        data = yaml.safe_load(f)
    validator = MappingValidator(data)
    is_valid = validator.validate()
    assert is_valid, f"TPU mapping YAML validation errors: {validator.errors}"
