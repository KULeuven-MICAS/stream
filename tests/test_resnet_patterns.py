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
