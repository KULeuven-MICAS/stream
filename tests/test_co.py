"""End-to-end test for the two-conv TPU constraint optimization pipeline."""

from pathlib import Path

import pytest

from stream.api import optimize_allocation_co
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import (
    TwoConvWorkloadConfig,
    make_2_conv_workload,
)
from stream.workload.node import ComputationNode

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


@pytest.fixture
def output_dir(request, tmp_path: Path):
    """Provide an output directory for the test.

    Normal CI runs:
        Uses a temporary directory automatically cleaned by pytest.

    Debug runs with:
        pytest --keep-output

    will instead store outputs in:
        outputs/test_co_tpu_two_conv
    """
    keep = request.config.getoption("--keep-output")

    if keep:
        path = Path("outputs/test_co_tpu_two_conv")
        path.mkdir(parents=True, exist_ok=True)

        print(f"\nKeeping outputs in: {path.resolve()}")

        return path

    path = tmp_path / "outputs"
    path.mkdir()

    print(f"\nTemporary outputs in: {path}")

    return path


def test_co_tpu_two_conv(output_dir: Path):
    """Run the two-conv TPU CO pipeline and verify structural properties.

    Asserts:
    - positive scheduler metrics
    - exactly two computation nodes
    - each computation node has a resource allocation
    """
    workload_path = make_2_conv_workload(_WORKLOAD_CONFIG)
    mapping_path = make_2_conv_mapping(_WORKLOAD_CONFIG)

    print(f"Workload path: {workload_path}")
    print(f"Mapping path: {mapping_path}")

    ctx = optimize_allocation_co(
        hardware=_ACCELERATOR,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id="test-tpu-two-conv",
        output_path=str(output_dir),
        skip_if_exists=False,
    )

    scheduler: SteadyStateScheduler = ctx.get("scheduler")

    print(f"latency_total: {scheduler.latency_total}")
    print(f"latency_per_iteration: {scheduler.latency_per_iteration}")
    print(f"iterations: {scheduler.iterations}")

    assert scheduler.latency_total > 0, "Expected positive latency_total"
    assert scheduler.latency_per_iteration > 0, "Expected positive latency_per_iteration"
    assert scheduler.iterations > 0, "Expected positive iterations"

    mapping = ctx.get("mapping")
    workload = ctx.get("workload")

    computation_nodes = [n for n in workload.nodes if isinstance(n, ComputationNode)]

    assert len(computation_nodes) == 2, f"Expected 2 computation nodes, got {len(computation_nodes)}"

    for node in computation_nodes:
        nm = mapping.get(node)

        print(f"Node {node.name}: {nm.resource_allocation}")

        assert nm.resource_allocation, f"Node {node.name} has empty resource_allocation"
