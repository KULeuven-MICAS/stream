"""The steady-state IR states what the solver solved: loops outermost first, absent loops counted, integers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conv_windows import conv, onnx_graph
from onnx import helper

from stream.allocation.allocation import Allocation
from stream.api import SolveOptions, evaluate_mapping
from stream.ir import AllocationIR
from stream.opt.allocation.constraint_optimization.allocation_model import solved_latency

HARDWARE = "stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml"
CHAIN_MAPPING = """
layers:
- {name: Conv1, core_allocation: [[0]]}
- {name: Conv2, core_allocation: [[0]]}
fused_groups:
- {name: g, layers: [Conv1, Conv2], intra_core_tiling: [{dim: Conv2.D6, tile: 2}, {dim: Conv1.D2, tile: 4}]}
"""
SPLIT_MAPPING = """
layers:
- {name: Conv1, core_allocation: [[0, 1, 2, 3]], inter_core_tiling: [[{dim: D2, split: 4}]]}
- {name: Conv2, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D2, split: 2}]]}
fused_groups:
- {name: g, layers: [Conv1, Conv2], intra_core_tiling: [{dim: Conv1.D1, tile: 4}]}
"""
FORK_MAPPING = """
layers:
- {name: ConvA, core_allocation: [[0]]}
- {name: ConvB, core_allocation: [[0]]}
- {name: Add, core_allocation: [[0]]}
fused_groups:
- {name: g, layers: [ConvA, ConvB, Add], intra_core_tiling: [{dim: ConvA.D2, tile: 4}]}
"""


def _two_convs() -> str:
    """input (1, 4, 16, 8) -> Conv1 (4, 3x3) -> Conv2 (4, 3x3), both padded by 1."""
    return onnx_graph(
        [conv("Conv1", "input", "w1", "conv1_out"), conv("Conv2", "conv1_out", "w2", "out")],
        {"input": [1, 4, 16, 8], "w1": [4, 4, 3, 3], "w2": [4, 4, 3, 3]},
        ["out"],
    )


def _solve(tmp_path: Path, workload: str, mapping: str) -> tuple[Allocation, dict[str, Any]]:
    (tmp_path / "mapping.yaml").write_text(mapping)
    options = SolveOptions(artifacts=False)
    estimate = evaluate_mapping(HARDWARE, workload, str(tmp_path), str(tmp_path / "mapping.yaml"), options)
    allocation = estimate.context.get("allocation")
    return allocation, AllocationIR.from_internal(allocation).model_dump()


def _group_loops(ir: dict[str, Any]) -> list[tuple]:
    return [(loop["size"], loop["type"]) for loop in ir["steady_state"]["loops"] if loop["node"] is None]


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> tuple[Allocation, dict[str, Any]]:
    """The two convs on one core: the 16 rows in 4 tiles of 4 (outer), Conv2's 4 output channels in 2 tiles of 2
    (inner), 8 iterations. Conv1 does not index Conv2's output channels, so it is absent from the inner loop."""
    return _solve(tmp_path_factory.mktemp("chain"), _two_convs(), CHAIN_MAPPING)


def test_the_group_loops_run_outermost_first(chain):
    """The rows loop (4 trips) comes first and the channel loop (2 trips) second."""
    assert _group_loops(chain[1]) == [(4, "temporal"), (2, "temporal")]


def test_a_dim_two_nodes_split_differently_is_one_group_loop_at_the_larger_split(tmp_path: Path):
    """Conv1 splits the 16 rows over 4 cores, Conv2 over 2: one spatial loop of 4 below the 2 fused column tiles."""
    _, ir = _solve(tmp_path, _two_convs(), SPLIT_MAPPING)
    assert _group_loops(ir) == [(2, "temporal"), (4, "spatial")]


def test_a_node_and_a_transfer_absent_from_a_loop_count_its_trips(chain):
    """Conv1 runs once per row tile, 8 / 2 = 4 times. Its input fires once per row tile too, each firing serving the
    2 channel trips: reuse factor 2 in the node, the tensor reuse report and the slot breakdown."""
    allocation, ir = chain
    nodes = {node["name"]: node for node in ir["steady_state"]["tiled_graph"]["nodes"]}
    assert ir["steady_state"]["iterations"] == 8
    assert (nodes["Conv1"]["calls"], nodes["Conv2"]["calls"]) == (4, 8)
    load = nodes["Transfer(input)"]
    assert load["reuse_factor"] == 2
    assert [(loop["size"], loop["effect"]) for loop in load["loops"] if loop["type"] == "temporal"] == [
        (4, "varying"),
        (2, "absent"),
    ]
    reuse = {row["tensor"]: row["reuse_factor"] for row in ir["performance"]["tensor_reuse"]}
    assert reuse["input_1"] == 2
    rows = [
        row for slot in allocation.solution.slot_latency_breakdown["slots"] for row in slot["transfer_contributors"]
    ]
    assert next(row["reuse_factor"] for row in rows if row["transfer"] == "Transfer(input)") == 2


def test_a_copy_is_the_window_its_reader_holds(chain):
    """Conv2 reads 4 output rows plus a row either side of conv1_out: (1, 4, 6, 8) at 16 bits, 3072 bits."""
    copy = next(t for t in chain[1]["steady_state"]["tensors"] if t["name"] == "conv1_out_1")
    assert (copy["shape"], copy["bits"]) == ([1, 4, 6, 8], 3072)


def test_each_copy_reports_its_own_halo(tmp_path: Path):
    """A 3x3 and a 5x5 convolution read one input tiled by 4 rows: copy 0 (ConvA's) keeps 3 - 1 = 2 rows across a
    row step, copy 1 (ConvB's) 5 - 1 = 4; each target's window is its copy's 4 rows plus that halo."""
    workload = onnx_graph(
        [
            conv("ConvA", "input", "wa", "a"),
            helper.make_node("Conv", ["input", "wb"], ["b"], name="ConvB", kernel_shape=[5, 5], pads=[2] * 4),
            helper.make_node("Add", ["a", "b"], ["o"], name="Add"),
        ],
        {"input": [1, 2, 8, 4], "wa": [2, 2, 3, 3], "wb": [2, 2, 5, 5]},
        ["o"],
    )
    _, ir = _solve(tmp_path, workload, FORK_MAPPING)
    load = next(node for node in ir["steady_state"]["tiled_graph"]["nodes"] if node["name"] == "Transfer(input)")
    halos = {(halo["copy"], halo["type"]): halo["halo"] for halo in load["halos"]}
    assert halos == {(0, "temporal"): 2, (1, "temporal"): 4}
    assert [window[2] for window in load["windows"]] == [4 + halos[(c, "temporal")] for c in load["target_copies"]]


def test_the_latency_reads_back_integers_that_keep_the_identity():
    """Two slots the solver returns as 295.9999 cycles are 296 each and an overlap of 591.9999 is 592: 8 iterations
    take 8 x 592 - 7 x 592 + 3 = 595 cycles."""
    latency = solved_latency(594.9999, {0: 295.9999, 1: 295.9999}, 591.9999, 3.0001)
    assert (latency.total, latency.per_iteration, latency.overlap, latency.fill) == (595, 592, 592, 3)
    assert latency.total == 8 * latency.per_iteration - 7 * latency.overlap + latency.fill
