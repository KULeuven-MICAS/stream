"""The memory-ports family on the example ZigZag accelerators, against the prototype that derived it."""

import tempfile
from collections.abc import Callable
from typing import Any

import pytest

from stream.api import SolveOptions
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta

HARDWARE = "stream/inputs/examples/hardware/{}.yaml"
SWIGLU_TILING = [
    {"dim": "Gemm_Left.D1", "tile": 128},
    {"dim": "Gemm_Down.D2", "tile": 128},
    {"dim": "Gemm_Left.D2", "tile": 32},
    {"dim": "Gemm_Left.D0", "tile": 16},
]

# Total latency and busiest (memory, port): the prototype's (evidence_59a9340/ports.jsonl), and the family's output
# at merge. `pytest -m slow` on this file prints the current values of a case that changes.
TWO_CONV_CASES = [
    ("eyeriss_like_single_core", 114999, ("dram", "rw_port_1")),
    ("eyeriss_like_dual_core", 80915, ("sram_1M", "rw_port_2")),
    ("eyeriss_like_quad_core", 46031, ("sram_1M", "rw_port_2")),
    ("tpu_like_quad_core", 18072, ("dram", "rw_port_1")),
    ("simba_small", 14913, ("dram", "rw_port_1")),
    ("simba", 12308, ("dram", "rw_port_1")),
    ("fusemax", 185892, ("sram", "r_port_1")),
    ("meta_prototype_dual_core_simd_offchip", 21625, ("dram", "rw_port_1")),
]
SWIGLU_CASES = [
    ("eyeriss_like_single_core", 303038597, ("sram_64KB", "r_port_1")),
    ("eyeriss_like_dual_core", 201589115, ("dram", "rw_port_1")),
    ("eyeriss_like_quad_core", 201589494, ("dram", "rw_port_1")),
    ("tpu_like_quad_core", 201588814, ("dram", "rw_port_1")),
    ("simba_small", 201589342, ("dram", "rw_port_1")),
    # One cycle above the prototype, which dropped streams whose active latency rounds to 0.
    ("simba", 201589015, ("dram", "rw_port_1")),
    ("fusemax", 76808228, ("sram", "r_port_1")),  # as simba
    ("meta_prototype_dual_core_simd_offchip", 201588793, ("dram", "rw_port_1")),
]


def solve_with_ports(solved_allocator: Callable[..., Any], hardware: str, workload: Any, **options: Any) -> Any:
    with tempfile.TemporaryDirectory() as out:
        solve_options = SolveOptions(families=["memory_ports"], **options)
        return solved_allocator(HARDWARE.format(hardware), workload, out, options=solve_options)


def port_activity(alloc: tta.TransferAndTensorAllocator) -> list[dict]:
    """The port report, after checking that no port moves more than its bandwidth allows."""
    rows = alloc.compute_performance_stats()["memory_ports"]
    assert all(row["stall_or_slack"] <= 1e-6 for row in rows)
    assert all((row["burst_utilization"] or 0.0) <= 1.0 + 1e-9 for row in rows)
    return rows


@pytest.mark.slow
@pytest.mark.parametrize(("hardware", "latency", "port"), TWO_CONV_CASES, ids=[c[0] for c in TWO_CONV_CASES])
def test_two_conv_matches_the_prototype(
    solved_allocator: Callable,
    two_conv: TwoConvWorkloadConfig,
    hardware: str,
    latency: int,
    port: tuple[str, str],
) -> None:
    alloc = solve_with_ports(solved_allocator, hardware, make_2_conv_workload(two_conv))
    assert alloc.total_latency.X == latency
    assert port_activity(alloc)[0]["port"] == ".".join(port)


@pytest.mark.slow
@pytest.mark.parametrize(("hardware", "latency", "port"), SWIGLU_CASES, ids=[c[0] for c in SWIGLU_CASES])
def test_fused_swiglu_matches_the_prototype(
    solved_allocator: Callable, hardware: str, latency: int, port: tuple[str, str]
) -> None:
    workload = make_small_swiglu_workload(seq_len=256, embedding_dim=2048, hidden_dim=8192)
    tiling = {"intra_core_tiling": SWIGLU_TILING}
    alloc = solve_with_ports(solved_allocator, hardware, workload, stage_options=tiling)
    assert alloc.total_latency.X == latency
    busiest = port_activity(alloc)[0]
    assert busiest["port"] == ".".join(port)
    # Every weight is re-read once per sequence tile, so wherever DRAM is busiest it sets the interval.
    if port[0] == "dram":
        assert busiest["utilization"] == pytest.approx(1.0, abs=1e-3)
