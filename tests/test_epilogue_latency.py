"""A reduction held over every iteration fires once, after the last: the run's epilogue times it, not the iterations."""

import tempfile
from pathlib import Path

from conv_windows import HARDWARE, onnx_graph
from onnx import helper

from stream.api import SolveOptions, evaluate_mapping

ONE_CORE = """layers:
- {name: Gemm, core_allocation: [[0]]}
fused_groups:
- {name: G, layers: [Gemm], intra_core_tiling: [{dim: Gemm.D2, tile: 16}]}
"""

CONTRACTION_TILED = """layers:
- {name: Gemm, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D2, split: 2}]]}
fused_groups:
- {name: G, layers: [Gemm], intra_core_tiling: [{dim: Gemm.D2, tile: 16}]}
"""


def _solve(shapes: dict[str, list[int]], mapping: str):
    """The estimate of Y = X W on tpu_like_quad_core under ``mapping``."""
    gemm = helper.make_node("MatMul", ["X", "W"], ["Y"], name="Gemm")
    workload = onnx_graph([gemm], shapes, ["Y"])
    with tempfile.TemporaryDirectory() as out:
        path = Path(out) / "mapping.yaml"
        path.write_text(mapping)
        return evaluate_mapping(
            HARDWARE.format("tpu_like_quad_core"), workload, out, str(path), SolveOptions(artifacts=False)
        )


def test_a_sum_written_off_chip_once_takes_the_ports_time_after_the_last_iteration():
    """Y = X W, each 64 x 64 bf16 (65536 bits), on one core, the contraction walked in four tiles of 16: Y is held over
    the four iterations and written off chip once, after the last. The off-chip port passes 64 bits a cycle, so that
    firing takes 65536 / 64 = 1024 cycles, the epilogue; per iteration the port moves a quarter of X and W, 32768
    bits."""
    estimate = _solve({"X": [64, 64], "W": [64, 64]}, ONE_CORE)
    solution = estimate.context.get("allocation").solution
    assert solution.slot_latency_breakdown["totals"]["epilogue_cycles"] == 1024
    (port,) = (r for r in solution.performance["memory_ports"] if r["resource"] == "offchip.rw_port_1")
    assert (port["bits_per_iteration"], port["once_per_run_bits"], port["utilization"]) == (32768, 65536, 1)
    (group,) = estimate.context.data["group_allocations"].values()
    (level,) = (lv for lv in group["roofline"]["levels"] if lv["name"] == "off-chip memory")
    assert level["bytes"] == 3 * 65536 / 8
    assert estimate.cycles >= 3 * 65536 / 64


def test_a_reduction_over_cores_fired_once_is_the_runs_epilogue():
    """Y = X W with X (16,64), W (64,32), the contraction split over cores 0 and 1 and each core's 32 walked in two
    tiles of 16: two iterations, each core's partial sum of all of Y held over both. It is reduce-scattered once,
    after the last iteration's Gemm, each core sending the other half at bf16 and its finished half off chip:
    4*4096 = 16384 bits on the 128-bit bus, 128 cycles. Its slot holds nothing of it, and the run ends 128 cycles
    after the last iteration and its drain."""
    estimate = _solve({"X": [16, 64], "W": [64, 32]}, CONTRACTION_TILED)
    solution = estimate.context.get("allocation").solution
    breakdown = solution.slot_latency_breakdown
    (slot,) = (s for s in breakdown["slots"] for c in s["transfer_contributors"] if c["transfer"] == "Transfer(Y_1)")
    (reduction,) = slot["transfer_contributors"]
    assert (reduction["path_cycles"], reduction["reuse_factor"]) == (128, 2)
    assert slot["slot_latency_cycles"] == reduction["latency_contribution_cycles"] == 0
    assert breakdown["totals"]["epilogue_cycles"] == 128
    latency = solution.latency
    assert latency.total == 2 * latency.per_iteration - latency.overlap + latency.fill
    assert latency.fill >= 128
