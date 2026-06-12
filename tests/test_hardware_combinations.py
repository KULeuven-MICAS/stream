"""Parametrized CO pipeline tests across non-AIE example hardware.

Matrix: 8 non-AIE hardware x {2-conv, swiglu} = 16 CO-pipeline combinations — all green.
Parse check: test_hardware_parses covers all 8 architectures (HWFIX-05).
The 2-conv arm is a small workload with generic auto-tiling. The swiglu arm is a realistically-sized,
layer-fused FFN block (seq_len=256, embedding_dim=2048, hidden_dim=8192) driven with the justfile's
fused intra-core tiling (seq=16/embedding=128/hidden=32), so the solver costs one steady-state tile
rather than the full layer and stays fast (~6-10s per hardware, including the 36-core simba mesh).
No xfail/skip and no slow marks: every combination runs in the default fast suite.

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
# SwiGLU CO config — dims and fused intra-core tiles mirror the AIE `just swiglu` recipe, so the CO
# matrix exercises a realistically-sized, layer-fused FFN block. The fused intra-core tiling makes
# the solver cost ONE steady-state tile (then x iterations), keeping the big workload tractable.
# ---------------------------------------------------------------------------

_SWIGLU_SEQ_LEN = 256
_SWIGLU_EMBEDDING_DIM = 2048
_SWIGLU_HIDDEN_DIM = 8192
_SWIGLU_SEQ_TILE = 16
_SWIGLU_EMBEDDING_TILE = 128
_SWIGLU_HIDDEN_TILE = 32

# Fused-group intra-core tiling. Gemm_Left dims: D0=seq, D1=embedding, D2=hidden; Gemm_Down: D2=embedding.
# seq (D0) and hidden (D2 of the branch gemms) are shared across the fused group; embedding is the
# reduction dim of the branch gemms (Gemm_Left.D1) and the output dim of the down-proj (Gemm_Down.D2).
_SWIGLU_INTRA_CORE_TILING = [
    {"dim": "Gemm_Left.D1", "tile": _SWIGLU_EMBEDDING_TILE},  # embedding (K of left/right gemm)
    {"dim": "Gemm_Down.D2", "tile": _SWIGLU_EMBEDDING_TILE},  # embedding (N of down-proj gemm)
    {"dim": "Gemm_Left.D2", "tile": _SWIGLU_HIDDEN_TILE},  # hidden (N of left/right gemm)
    {"dim": "Gemm_Left.D0", "tile": _SWIGLU_SEQ_TILE},  # seq_len (M, shared across the group)
]

# Human-readable hyperparameter summaries surfaced in the CI metrics comment (per-workload caption).
_TWO_CONV_HPARAMS = (
    f"batch={_SMALL_2CONV_CONFIG.batch_size}, in_ch={_SMALL_2CONV_CONFIG.in_channels}, "
    f"H={_SMALL_2CONV_CONFIG.height}, W={_SMALL_2CONV_CONFIG.width}, "
    f"out_ch1={_SMALL_2CONV_CONFIG.out_channels_1}, out_ch2={_SMALL_2CONV_CONFIG.out_channels_2}, "
    f"kernel={_SMALL_2CONV_CONFIG.kernel_size}x{_SMALL_2CONV_CONFIG.kernel_size}, "
    f"{_SMALL_2CONV_CONFIG.in_dtype} (generic auto-tiling, not layer-fused)"
)
_SWIGLU_HPARAMS = (
    f"seq_len={_SWIGLU_SEQ_LEN}, embedding_dim={_SWIGLU_EMBEDDING_DIM}, hidden_dim={_SWIGLU_HIDDEN_DIM}, bf16; "
    f"layer-fused tiles seq={_SWIGLU_SEQ_TILE}/embedding={_SWIGLU_EMBEDDING_TILE}/hidden={_SWIGLU_HIDDEN_TILE}"
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
# Metrics capture helper — shared by the two_conv and swiglu arms
# ---------------------------------------------------------------------------


def _record_co_metrics(record_metric, ctx, workload_hparams: str | None = None) -> None:
    """Capture advisory CO metrics from the solved context (read-only side-effect, Phase 39).

    Beyond the gated total_latency, records observability metrics derived from the solved
    schedule's performance summary so the CI comment can explain outliers:
      - mac_spatial_utilization: latency-weighted MAC spatial utilization (how full the array is)
      - end_to_end_mac_utilization: useful MACs / (chip peak MACs/cycle x total latency), i.e. the
        true fraction of the chip's compute used (folds in idle cores, temporal stalls, transfers)
      - degenerate: True iff a matmul/conv node fell back to ZigZag's 1-MAC/cycle scalar cost,
        i.e. the spatial array was not modelled and the latency is untrustworthy.
      - workload_hparams: human-readable dims/tiles of this combination (per-workload caption).
    """
    scheduler = ctx.get("scheduler")
    solve_stats = scheduler.solve_stats if scheduler is not None else None
    perf = scheduler.performance_stats if scheduler is not None else None
    agg = perf.get("aggregate") if isinstance(perf, dict) else None
    group_latencies = ctx.get("group_latencies")
    record_metric("total_latency", ctx.get("total_latency"))
    record_metric("group_latencies_max", max(group_latencies.values()) if group_latencies else None)
    record_metric("objective", solve_stats.objective if solve_stats is not None else None)
    record_metric("mip_gap", solve_stats.mip_gap if solve_stats is not None else None)
    record_metric("solve_time_s", solve_stats.solve_time_s if solve_stats is not None else None)
    record_metric("mac_spatial_utilization", agg.get("latency_weighted_mac_spatial_utilization") if agg else None)
    record_metric("end_to_end_mac_utilization", agg.get("end_to_end_mac_utilization") if agg else None)
    record_metric("degenerate", agg.get("degenerate") if agg else None)
    record_metric("workload_hparams", workload_hparams)


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
    _record_co_metrics(record_metric, ctx, _TWO_CONV_HPARAMS)


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_hardware_swiglu(hardware: str, record_metric) -> None:
    """Run the layer-fused SwiGLU through the generic CO pipeline on each hardware.

    Selected by: pytest -k swiglu
    Uses the justfile `just swiglu` dims (seq_len=256, embedding_dim=2048, hidden_dim=8192) and the
    fused intra-core tiling (seq=16/embedding=128/hidden=32) so the whole 5-node block is processed
    layer-fused: the solver costs ONE steady-state tile and multiplies by the iteration count.
    Exercises HWFIX-04: Silu and Mul ops dispatch (to the simd core where available, or a
    generic compute core) rather than failing at the parser or mapper stage.
    """
    # 5-node count confirmed by running: Gemm_Left, Gemm_Right, Silu, Elt_Mul, Gemm_Down
    workload_path = make_small_swiglu_workload(
        seq_len=_SWIGLU_SEQ_LEN,
        embedding_dim=_SWIGLU_EMBEDDING_DIM,
        hidden_dim=_SWIGLU_HIDDEN_DIM,
    )
    hw_stem = Path(hardware).stem
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=hardware,
            workload=workload_path,
            experiment_id=f"swiglu_{hw_stem}",
            output_path=tmpdir,
            intra_core_tiling=_SWIGLU_INTRA_CORE_TILING,
        )
    accelerator = ctx.get("accelerator")
    _assert_co_result(ctx, accelerator, expected_node_count=5)
    # metrics capture — read-only advisory side-effect (Phase 39 CAP-01/CAP-02)
    _record_co_metrics(record_metric, ctx, _SWIGLU_HPARAMS)


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
