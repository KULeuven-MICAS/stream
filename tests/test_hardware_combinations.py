"""Parametrized CO pipeline tests across non-AIE example hardware.

Matrix: 8 non-AIE hardware x {2-conv, swiglu} = 16 CO-pipeline combinations — all green.
Parse check: test_hardware_parses covers all 8 architectures (HWFIX-05).
No xfail/skip and no slow marks: every combination runs in the default fast suite. The 36-core
simba mesh is included — on these small single-fusion-group workloads it runs in ~16s (2-conv) /
~7s (swiglu), on par with the other architectures. (Multi-fusion-group workloads, where the CO runs
once per group, are where large meshes get expensive — none of those live here.)

Fast suite (default pytest): 24 tests from this file (16 CO + 8 parse), none deselected.

Background (Phase 32): example accelerators referenced cores by bare filename; core resolution
used an ambiguous input-tree search that loaded testing core files (lacking operator_types) instead
of examples ones. Fix: accelerator-local core resolution + correct elementwise operator_types on
simd. Later work brought stale YAML definitions current (simba, fusemax, meta_prototype + simd
Silu/Mul) and added the swiglu workload builder and swiglu arm.
"""

import tempfile
from pathlib import Path

import pytest
from zigzag.utils import open_yaml

from stream.api import optimize_allocation_co_generic
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.node import ComputationNode

# ---------------------------------------------------------------------------
# Hardware under test — the 8 non-AIE example architectures. Append pytest.param entries
# to extend; use xfail/skip (with a written reason) only for a genuinely-infeasible
# workload x hardware combination, never to hide a fixable failure.
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
        # Not slow-marked: on these small single-fusion-group workloads the 36-core case
        # runs in ~16s (2-conv) / ~7s (swiglu) — on par with simba_small and the other architectures.
        "stream/inputs/examples/hardware/simba.yaml",
        id="simba",
    ),
    pytest.param(
        "stream/inputs/examples/hardware/fusemax.yaml",
        id="fusemax",
    ),
    pytest.param(
        "stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml",
        id="meta_prototype",
    ),
]

# All 8 architecture paths as plain strings (NO pytest marks). Separate from _HARDWARE so that simba's
# `slow` mark is NOT inherited — parsing is cheap (<1s), only simba's 36-core MILP is slow, so all
# 8 parse-checks must run in the fast suite (HWFIX-05).
_ALL_HARDWARE_PATHS = [
    "stream/inputs/examples/hardware/eyeriss_like_single_core.yaml",
    "stream/inputs/examples/hardware/eyeriss_like_dual_core.yaml",
    "stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml",
    "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
    "stream/inputs/examples/hardware/simba_small.yaml",
    "stream/inputs/examples/hardware/simba.yaml",
    "stream/inputs/examples/hardware/fusemax.yaml",
    "stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml",
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
def test_hardware_two_conv(hardware: str, record_metric) -> None:
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
    # metrics capture — read-only advisory side-effect (Phase 39 CAP-01/CAP-02)
    scheduler = ctx.get("scheduler")
    solve_stats = scheduler.solve_stats if scheduler is not None else None
    record_metric("total_latency", ctx.get("total_latency"))
    record_metric(
        "group_latencies_max",
        max(ctx.get("group_latencies").values()) if ctx.get("group_latencies") else None,
    )
    record_metric("objective", solve_stats.objective if solve_stats is not None else None)
    record_metric("mip_gap", solve_stats.mip_gap if solve_stats is not None else None)
    record_metric("solve_time_s", solve_stats.solve_time_s if solve_stats is not None else None)


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_hardware_swiglu_small(hardware: str, record_metric) -> None:
    """Run the swiglu workload through the generic CO pipeline on each hardware.

    Selected by: pytest -k swiglu
    Exercises HWFIX-04: Silu and Mul ops dispatch (to the simd core where available, or a
    generic compute core) rather than failing at the parser or mapper stage.
    """
    # 5-node count confirmed by running: Gemm_Left, Gemm_Right, Silu, Elt_Mul, Gemm_Down
    workload_path = make_small_swiglu_workload()
    hw_stem = Path(hardware).stem
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=hardware,
            workload=workload_path,
            experiment_id=f"swiglu_{hw_stem}",
            output_path=tmpdir,
        )
    accelerator = ctx.get("accelerator")
    _assert_co_result(ctx, accelerator, expected_node_count=5)
    # metrics capture — read-only advisory side-effect (Phase 39 CAP-01/CAP-02)
    scheduler = ctx.get("scheduler")
    solve_stats = scheduler.solve_stats if scheduler is not None else None
    record_metric("total_latency", ctx.get("total_latency"))
    record_metric(
        "group_latencies_max",
        max(ctx.get("group_latencies").values()) if ctx.get("group_latencies") else None,
    )
    record_metric("objective", solve_stats.objective if solve_stats is not None else None)
    record_metric("mip_gap", solve_stats.mip_gap if solve_stats is not None else None)
    record_metric("solve_time_s", solve_stats.solve_time_s if solve_stats is not None else None)


@pytest.mark.parametrize("path", _ALL_HARDWARE_PATHS)
def test_hardware_parses(path: str) -> None:
    """All 8 non-AIE hardware definitions parse, validate, and load without error (HWFIX-05).

    Uses the validator + factory load chain directly (open_yaml -> validate -> create) rather than
    the pipeline parser stage, which requires a pre-populated StageContext. Simba is included
    without a slow mark: parsing is cheap (<1s); only the 36-core MILP is slow.
    """
    data = open_yaml(path)
    validator = AcceleratorValidator(data, path)
    normalized = validator.normalized_data
    validate_ok = validator.validate()
    assert validate_ok, f"AcceleratorValidator.validate() returned False for {path}"
    accelerator = AcceleratorFactory(normalized).create()
    assert accelerator is not None
