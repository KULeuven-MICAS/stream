"""End-to-end test for the two-conv TPU constraint optimization pipeline."""

from pathlib import Path

import pytest

from stream.allocation.allocation import Allocation
from stream.api import evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import (
    TwoConvWorkloadConfig,
    make_2_conv_workload,
)
from stream.ir.allocation import AllocationIR
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
    - positive schedule metrics
    - exactly two computation nodes
    - each computation node has a resource allocation
    """
    workload_path = make_2_conv_workload(_WORKLOAD_CONFIG)
    mapping_path = make_2_conv_mapping(_WORKLOAD_CONFIG)

    print(f"Workload path: {workload_path}")
    print(f"Mapping path: {mapping_path}")

    ctx = evaluate_mapping(_ACCELERATOR, workload_path, str(output_dir), mapping_path).context

    schedule: Allocation = ctx.get("allocation")
    latency = schedule.solution.latency

    print(f"latency_total: {latency.total}")
    print(f"latency_per_iteration: {latency.per_iteration}")
    print(f"iterations: {schedule.iterations}")

    assert latency.total > 0, "Expected positive latency_total"
    assert latency.per_iteration > 0, "Expected positive latency_per_iteration"
    assert schedule.iterations > 0, "Expected positive iterations"
    # OR-Tools solves the objective levels in turn and ends on a tiebreaker; the rank is the first level.
    assert schedule.cost_to_rank >= latency.total

    # The solved schedule must yield a JSON-serializable AllocationIR (runtime_args carry AffineMaps).
    allocation = AllocationIR.from_internal(schedule)
    allocation.model_dump_json()
    assert allocation.latency.total == latency.total
    assert allocation.mapping_nodes

    mapping = ctx.get("mapping")
    workload = ctx.get("workload")

    computation_nodes = [n for n in workload.nodes if isinstance(n, ComputationNode)]

    assert len(computation_nodes) == 2, f"Expected 2 computation nodes, got {len(computation_nodes)}"

    for node in computation_nodes:
        nm = mapping.get(node)

        print(f"Node {node.name}: {nm.resource_allocation}")

        assert nm.resource_allocation, f"Node {node.name} has empty resource_allocation"
