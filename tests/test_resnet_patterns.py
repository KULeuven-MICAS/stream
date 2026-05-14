"""Integration tests for ResNet18 sub-graph patterns through the CO pipeline.

Tests RNET-01 (basic residual), RNET-02 (stride-2 downsample), RNET-03 (front-end + pooling core).
Also tests dual-residual multi-group split via identity Reshape FusionEdge.

Per D-05: dedicated test file separate from test_generic_mapping.py.
Per D-06: assertions check positive latency, correct group count, and core type allocation.
"""

import os
import tempfile

import pytest
import yaml

from stream.api import optimize_allocation_co_generic
from stream.inputs.testing.workload.make_resnet_subgraph import (
    ResNetPattern,
    ResNetSubgraphConfig,
    make_resnet_subgraph,
)

_ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


@pytest.mark.timeout(120)
def test_basic_residual():
    """RNET-01: Basic residual block (Conv->Relu->Conv->Add with skip fan-out)
    completes CO allocation with positive latency and 1 group."""
    config = ResNetSubgraphConfig(pattern=ResNetPattern.BASIC_RESIDUAL)
    onnx_path = make_resnet_subgraph(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=onnx_path,
            experiment_id="test-basic-residual",
            output_path=tmpdir,
        )

        total_latency = ctx.get("total_latency")
        assert total_latency is not None and total_latency > 0, f"Expected positive total_latency, got {total_latency}"
        group_latencies = ctx.get("group_latencies")
        assert group_latencies is not None, "group_latencies not set in context"
        assert len(group_latencies) == 1, f"Basic residual should have 1 group, got {len(group_latencies)}"
        assert all(lat > 0 for lat in group_latencies.values()), (
            f"All group latencies must be positive, got {group_latencies}"
        )


@pytest.mark.timeout(120)
def test_stride2_downsample():
    """RNET-02: Stride-2 residual with 1x1 downsample runs through full CO
    pipeline with positive latency and 1 group."""
    config = ResNetSubgraphConfig(pattern=ResNetPattern.STRIDE2_DOWNSAMPLE)
    onnx_path = make_resnet_subgraph(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=onnx_path,
            experiment_id="test-stride2-downsample",
            output_path=tmpdir,
        )

        total_latency = ctx.get("total_latency")
        assert total_latency is not None and total_latency > 0, f"Expected positive total_latency, got {total_latency}"
        group_latencies = ctx.get("group_latencies")
        assert group_latencies is not None, "group_latencies not set in context"
        assert len(group_latencies) == 1, f"Stride-2 downsample should have 1 group, got {len(group_latencies)}"
        assert all(lat > 0 for lat in group_latencies.values()), (
            f"All group latencies must be positive, got {group_latencies}"
        )


@pytest.mark.timeout(120)
def test_frontend_path():
    """RNET-03: Front-end path (Conv(7x7,s=2)->Relu->MaxPool(s=2)) completes CO
    with MaxPool allocated to pooling core (core 4)."""
    config = ResNetSubgraphConfig(pattern=ResNetPattern.FRONTEND)
    onnx_path = make_resnet_subgraph(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=onnx_path,
            experiment_id="test-frontend-path",
            output_path=tmpdir,
        )

        # Latency assertions
        total_latency = ctx.get("total_latency")
        assert total_latency is not None and total_latency > 0, f"Expected positive total_latency, got {total_latency}"
        group_latencies = ctx.get("group_latencies")
        assert group_latencies is not None, "group_latencies not set in context"
        assert len(group_latencies) == 1, f"Frontend path should have 1 group, got {len(group_latencies)}"
        assert all(lat > 0 for lat in group_latencies.values()), (
            f"All group latencies must be positive, got {group_latencies}"
        )

        # RNET-03 core type assertion: MaxPool must be on pooling core (core 4)
        # Read the group mapping YAML from the output directory (still available inside `with`)
        experiment_dir = os.path.join(tmpdir, "test-frontend-path")
        group_dir = os.path.join(experiment_dir, "group_0")
        mapping_path = os.path.join(group_dir, "mapping.yaml")
        assert os.path.exists(mapping_path), f"Group mapping YAML not found at {mapping_path}"

        with open(mapping_path) as f:
            mapping_data = yaml.safe_load(f)

        # Find the MaxPool layer
        maxpool_layer = None
        for layer in mapping_data["layers"]:
            if "MaxPool" in layer["name"]:
                maxpool_layer = layer
                break
        assert maxpool_layer is not None, "MaxPool layer not found in mapping YAML"

        # Assert pooling core allocation (core 4)
        core_ids = [c for slot in maxpool_layer["core_allocation"] for c in slot]
        assert 4 in core_ids, f"MaxPool should be allocated to pooling core (id=4), got core_ids={core_ids}"


@pytest.mark.timeout(120)
def test_dual_residual():
    """Dual-residual (two blocks + Reshape FusionEdge) produces 2 fusion groups,
    both with positive latency."""
    config = ResNetSubgraphConfig(pattern=ResNetPattern.DUAL_RESIDUAL)
    onnx_path = make_resnet_subgraph(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ACCELERATOR,
            workload=onnx_path,
            experiment_id="test-dual-residual",
            output_path=tmpdir,
        )

        total_latency = ctx.get("total_latency")
        assert total_latency is not None and total_latency > 0, f"Expected positive total_latency, got {total_latency}"
        group_latencies = ctx.get("group_latencies")
        assert group_latencies is not None, "group_latencies not set in context"
        assert len(group_latencies) == 2, f"Dual-residual should have 2 groups, got {len(group_latencies)}"
        assert all(lat > 0 for lat in group_latencies.values()), (
            f"All group latencies must be positive, got {group_latencies}"
        )
        # Aggregation check: total = sum of per-group
        assert abs(total_latency - sum(group_latencies.values())) < 1e-6, (
            f"total_latency ({total_latency}) != sum of group_latencies ({sum(group_latencies.values())})"
        )


@pytest.mark.timeout(60)
def test_fusion_cut_points_heuristic():
    """RNET-04: determine_fusion_cut_points() returns correct cut-point names for ResNet18.

    Expected: 1 MaxPool + 8 Add+Relu = 9 cut points total.
    """
    from stream.parser.onnx.model import ONNXModelParser
    from stream.workload.workload import determine_fusion_cut_points

    parser = ONNXModelParser("stream/inputs/examples/workload/resnet18.onnx")
    parser.run()
    workload = parser.workload

    cut_points = determine_fusion_cut_points(workload)

    # Exactly 9 cut points: 1 MaxPool + 8 Relu (after Add)
    assert len(cut_points) == 9, f"Expected 9 cut points, got {len(cut_points)}: {cut_points}"

    # First cut point is the MaxPool (front-end boundary)
    assert "MaxPool" in cut_points[0], f"First cut point should be MaxPool, got {cut_points[0]}"

    # Remaining 8 are Relu nodes following Add nodes
    relu_cut_points = [cp for cp in cut_points if "Relu" in cp or "relu" in cp]
    assert len(relu_cut_points) == 8, f"Expected 8 Relu cut points, got {len(relu_cut_points)}"


@pytest.mark.timeout(60)
def test_resnet18_split_with_cut_points():
    """RNET-04: split_fusion_groups(cut_points=...) produces 11 groups for ResNet18.

    9 cut points + 1 FusionEdge (Flatten) = 10 boundaries -> 11 groups.
    """
    from stream.parser.onnx.model import ONNXModelParser
    from stream.workload.node import ComputationNode
    from stream.workload.workload import determine_fusion_cut_points

    parser = ONNXModelParser("stream/inputs/examples/workload/resnet18.onnx")
    parser.run()
    workload = parser.workload

    cut_points = determine_fusion_cut_points(workload)
    groups = workload.split_fusion_groups(cut_points=cut_points)

    assert len(groups) == 11, f"Expected 11 groups, got {len(groups)}"

    # Verify each group has at least 1 ComputationNode
    for i, group in enumerate(groups):
        comp_nodes = [n for n in group.nodes if isinstance(n, ComputationNode)]
        assert len(comp_nodes) >= 1, f"Group {i} has no ComputationNodes"
        # No single group should have more than 8 computation nodes
        assert len(comp_nodes) <= 8, f"Group {i} has {len(comp_nodes)} ComputationNodes (> 8)"

    # Group 0 (front-end): Conv + Relu + MaxPool = 3 ComputationNodes
    g0_comp = [n for n in groups[0].nodes if isinstance(n, ComputationNode)]
    assert len(g0_comp) == 3, f"Front-end group should have 3 nodes, got {len(g0_comp)}"

    # Last group (post-Flatten): just Gemm = 1 ComputationNode
    g_last_comp = [n for n in groups[-1].nodes if isinstance(n, ComputationNode)]
    assert len(g_last_comp) == 1, f"Last group should have 1 node (Gemm), got {len(g_last_comp)}"


@pytest.mark.timeout(120)
def test_resnet18_cut_point_groups():
    """RNET-05: Per-group mappings for ResNet18 (11 groups) all pass MappingValidator."""
    from zigzag.utils import open_yaml

    from stream.mapping.generic_generator import GenericMappingGenerator
    from stream.parser.accelerator_factory import AcceleratorFactory
    from stream.parser.accelerator_validator import AcceleratorValidator
    from stream.parser.mapping_validator import MappingValidator
    from stream.parser.onnx.model import ONNXModelParser
    from stream.workload.workload import determine_fusion_cut_points

    # Parse workload
    parser = ONNXModelParser("stream/inputs/examples/workload/resnet18.onnx")
    parser.run()
    workload = parser.workload

    # Parse accelerator (matching pipeline pattern: open_yaml -> validate -> factory.create)
    accel_data = open_yaml(_ACCELERATOR)
    validator = AcceleratorValidator(accel_data, _ACCELERATOR)
    accel_data = validator.normalized_data
    assert validator.validate(), f"Accelerator validation failed: {validator.errors}"
    factory = AcceleratorFactory(accel_data)
    accelerator = factory.create()

    # Determine cut points and generate per-group mappings
    cut_points = determine_fusion_cut_points(workload)
    assert len(cut_points) == 9

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = GenericMappingGenerator(
            accelerator=accelerator,
            workload=workload,
            output_dir=tmpdir,
        )
        paths, sub_workloads = generator.generate_all_groups(cut_points=cut_points)

        # 11 groups = 11 YAML paths
        assert len(paths) == 11, f"Expected 11 group mapping paths, got {len(paths)}"
        assert len(sub_workloads) == 11, f"Expected 11 sub-workloads, got {len(sub_workloads)}"

        # Each YAML must pass MappingValidator
        for i, path in enumerate(paths):
            assert os.path.exists(path), f"Group {i} mapping file not found: {path}"
            with open(path) as f:
                mapping_data = yaml.safe_load(f)
            mv = MappingValidator(mapping_data)
            assert mv.validate(), f"Group {i} mapping failed validation: {mv.errors}"
