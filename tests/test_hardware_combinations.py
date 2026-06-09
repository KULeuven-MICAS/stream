"""Parametrized CO pipeline tests across non-AIE example hardware.

Phase 32 covers 4 non-AIE hardware definitions; all run the 2-conv workload green via the
generic CO pipeline (eyeriss_like single/dual/quad + tpu_like_quad_core).

The initial run surfaced two failures on the eyeriss cores that turned out to share one
root cause: example accelerators reference cores by bare filename (e.g. ``pooling.yaml``),
and core resolution was an ambiguous input-tree search that loaded the *testing* core files
(which lack ``operator_types``) instead of the examples ones. With ``operator_types`` lost,
the generic mapper treated pooling+simd as generic compute cores and tiled Conv across all
non-offchip cores (6 on quad → ``32 % 6`` assert; over-subscription on single-core). The fix
(accelerator-local core resolution + correct elementwise ``operator_types`` on simd) makes
Conv tile across the real compute cores only, so all four pass.

Phases 33-35 append stale hardware (with optional slow marks) to _HARDWARE. Phase 36 adds a
swiglu test arm as a separate function reusing the same _assert_co_result helper.
"""

import tempfile
from pathlib import Path

import pytest

from stream.api import optimize_allocation_co_generic
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.workload.node import ComputationNode

# ---------------------------------------------------------------------------
# Hardware under test — extend in phases 33-35 (append pytest.param entries,
# with optional slow / xfail marks).
# ---------------------------------------------------------------------------

_HARDWARE = [
    pytest.param(
        "stream/inputs/examples/hardware/eyeriss_like_single_core.yaml",
        id="eyeriss_like_single_core",
    ),
    pytest.param(
        "stream/inputs/examples/hardware/eyeriss_like_dual_core.yaml",
        id="eyeriss_like_dual_core",
    ),
    pytest.param(
        "stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml",
        id="eyeriss_like_quad_core",
    ),
    pytest.param(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        id="tpu_like_quad_core",
    ),
    pytest.param(
        "stream/inputs/examples/hardware/simba_small.yaml",
        id="simba_small",
    ),
    pytest.param(
        # 36 compute chiplets: the generic mapper splits the 2-conv across dimensions
        # (e.g. OY x FY x FX = 4 x 3 x 3 = 36) since no single dim is divisible by 36.
        # slow-marked because the 36-core MILP is the heaviest case in the matrix.
        "stream/inputs/examples/hardware/simba.yaml",
        id="simba",
        marks=pytest.mark.slow,
    ),
]

_SMALL_2CONV_CONFIG = TwoConvWorkloadConfig(
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

# ---------------------------------------------------------------------------
# Shared assertion helper — reused by Phase 36 swiglu arm
# ---------------------------------------------------------------------------


def _assert_co_result(ctx, accelerator: Accelerator, expected_node_count: int) -> None:
    """Assert structural CO result properties.

    Reused by the Phase 36 swiglu arm. Checks:
    - Positive scheduler metrics (latency_total, latency_per_iteration, iterations)
    - Exactly expected_node_count ComputationNodes
    - Each node has non-empty resource_allocation
    - No ComputationNode allocated to the offchip core (HWTEST-03)
    """
    scheduler: SteadyStateScheduler = ctx.get("scheduler")
    assert scheduler.latency_total > 0, "Expected positive latency_total"
    assert scheduler.latency_per_iteration > 0, "Expected positive latency_per_iteration"
    assert scheduler.iterations > 0, "Expected positive iterations"

    mapping = ctx.get("mapping")
    workload = ctx.get("workload")
    computation_nodes = [n for n in workload.nodes if isinstance(n, ComputationNode)]

    assert len(computation_nodes) == expected_node_count, (
        f"Expected {expected_node_count} ComputationNodes, got {len(computation_nodes)}"
    )

    offchip_id = accelerator.offchip_core_id
    for node in computation_nodes:
        nm = mapping.get(node)
        assert nm.resource_allocation, f"Node {node.name} has empty resource_allocation"
        if offchip_id is not None:
            for core_group in nm.resource_allocation:
                for core in core_group:
                    assert core.id != offchip_id, (
                        f"ComputationNode {node.name} allocated to offchip core "
                        f"id={offchip_id} — possible degenerate type: compute on DRAM core definition"
                    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_hardware_two_conv(hardware: str) -> None:
    """Run the 2-conv workload through the generic CO pipeline on each green hardware.

    Selected by: pytest -k two_conv
    """
    workload_path = make_2_conv_workload(_SMALL_2CONV_CONFIG)
    hw_stem = Path(hardware).stem
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=hardware,
            workload=workload_path,
            experiment_id=f"two_conv_{hw_stem}",
            output_path=tmpdir,
        )
    accelerator = ctx.get("accelerator")
    _assert_co_result(ctx, accelerator, expected_node_count=2)
