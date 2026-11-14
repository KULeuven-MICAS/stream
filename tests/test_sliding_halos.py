"""Every loop that slides a copy's window keeps its halo: what a firing moves, what the run moves and what the copy
holds (bf16 elements, 16 bits each)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conv_windows import HARDWARE, conv, onnx_graph
from onnx import helper

from stream.allocation.allocation import Allocation
from stream.api import SolveOptions, evaluate_mapping
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.workload.node import TransferNode

EYERISS = HARDWARE.format("eyeriss_like_single_core")
SIMBA = HARDWARE.format("simba_small")
ROWS_OUTER = """
layers:
- {name: Conv1, core_allocation: [[0]]}
- {name: Conv2, core_allocation: [[0]]}
fused_groups:
- {name: g, layers: [Conv1, Conv2], intra_core_tiling: [{dim: Conv1.D1, tile: 2}, {dim: Conv1.D2, tile: 4}]}
"""
SINGLE = """
layers:
- {name: Conv1, core_allocation: [[0]]}
fused_groups:
- {name: g, layers: [Conv1], intra_core_tiling: [{dim: Conv1.D1, tile: 2}, {dim: Conv1.D2, tile: 4}]}
"""
RESIDUAL = """
layers:
- {name: Conv1, core_allocation: [[0]]}
- {name: Conv2, core_allocation: [[0]]}
- {name: Add1, core_allocation: [[0]]}
fused_groups:
- {name: g, layers: [Conv1, Conv2, Add1], intra_core_tiling: [{dim: Conv1.D1, tile: 8}, {dim: Conv1.D2, tile: 8}]}
"""


def _chain() -> str:
    """input (1,4,12,16) -> Conv1 (8, 3x3, pad 1) -> Conv2 (8, 3x3, pad 1), bf16."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out")]
    return onnx_graph(nodes, {"input": [1, 4, 12, 16], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


def _residual() -> str:
    """input (1,2,16,16) -> Conv1 (2, 3x3, pad 1) -> Conv2 (2, 3x3, pad 1) -> Add1 with input, bf16."""
    nodes = [
        conv("Conv1", "input", "w1", "c1"),
        conv("Conv2", "c1", "w2", "c2"),
        helper.make_node("Add", ["c2", "input"], ["out"], name="Add1"),
    ]
    return onnx_graph(nodes, {"input": [1, 2, 16, 16], "w1": [2, 2, 3, 3], "w2": [2, 2, 3, 3]}, ["out"])


def _solve(tmp_path: Path, hardware: str, workload: str, mapping: str) -> Allocation:
    (tmp_path / "mapping.yaml").write_text(mapping)
    options = SolveOptions(artifacts=False)
    estimate = evaluate_mapping(hardware, workload, str(tmp_path), str(tmp_path / "mapping.yaml"), options)
    return estimate.context.get("allocation")


def _into(allocation: Allocation, tensor: str) -> TransferNode:
    """The transfer moving ``tensor`` to its readers."""
    return next(tr for tr in allocation.problem.workload.get_transfer_nodes() if tr.inputs[0].name == tensor)


def _held(allocation: Allocation, name: str) -> int:
    """Bits the solved allocation holds of the tensor ``name`` over all memories."""
    rows = allocation.solution.performance["memory_occupancy"]
    return sum(entry["bits"] for row in rows for entry in row["tensors"] if entry["tensor"] == name)


@pytest.fixture(scope="module")
def rows_outer(tmp_path_factory: pytest.TempPathFactory) -> Allocation:
    """Columns in tiles of 2 (8 trips, inner) inside rows in tiles of 4 (3 trips, outer), 24 iterations."""
    return _solve(tmp_path_factory.mktemp("rows_outer"), EYERISS, _chain(), ROWS_OUTER)


def test_a_window_sliding_along_two_axes_moves_only_its_new_corner(rows_outer):
    """Conv1 makes 4 rows x 2 columns a firing from a 6 x 4 input window. The column loop keeps 2 columns, the row
    loop 2 rows: a firing moves 4 x 2 of the 4 channels, and the 24 firings move the input's 4 x 12 x 16 = 768
    elements once, 12288 bits."""
    workload, mapping, ssis = rows_outer.problem.workload, rows_outer.problem.mapping, rows_outer.problem.ssis
    tr = _into(rows_outer, "input")
    copy = tr.outputs[0]
    assert workload.get_tensor_of_transfer_to_single_core(copy, tr, mapping, ssis=ssis[copy]).shape == (1, 4, 4, 2)
    space = DecisionSpace(rows_outer.problem)
    assert space.moved_bits(tr, rows_outer.solution.transfer_routes[tr]) * space.iterations == 12288


def test_a_window_whose_halo_is_not_padding_moves_the_whole_input_once(tmp_path):
    """One unpadded 3x3 convolution making 12 x 16 output from a 4 x 14 x 18 input, tiled as above. The first trip of
    each loop brings its halo too: the 24 firings move every input element once, 4 x 14 x 18 x 16 bits."""
    shapes = {"input": [1, 4, 14, 18], "w1": [8, 4, 3, 3]}
    workload = onnx_graph([conv("Conv1", "input", "w1", "t1", pads=0)], shapes, ["t1"])
    allocation = _solve(tmp_path, EYERISS, workload, SINGLE)
    space, tr = DecisionSpace(allocation.problem), _into(allocation, "input")
    assert space.iterations == 24
    assert space.moved_bits(tr, allocation.solution.transfer_routes[tr]) * space.iterations == 4 * 14 * 18 * 16


def test_an_outer_sliding_loop_keeps_its_halo_across_the_inner_loops_trips(rows_outer):
    """Conv2 reads a 6 x 4 window of t1's 8 channels. The inner column loop's halo stays in the window; the row loop
    keeps 2 rows across the 16 columns the column loop sweeps, of which the window holds 4: 8 x 2 x 12 = 192 elements,
    3072 bits next to the tiles, none when the copy is kept across the column loop. Double-buffered,
    2 x 192 + 192 = 576 elements, 9216 bits of t1 in all."""
    space = DecisionSpace(rows_outer.problem)
    copy = _into(rows_outer, "t1").outputs[0]
    assert [space.halo_bits(copy, stop) for stop in space.stops(copy)] == [3072, 0, 0]
    assert _held(rows_outer, "t1") + _held(rows_outer, copy.name) == 9216


def test_copies_sliding_along_two_axes_together_move_each_element_once(tmp_path):
    """Conv1 and Add1 both read the input in 8 x 8 tiles of 16 x 16 (2 column trips inside 2 row trips), Conv1 a
    window of up to 10 x 10 with a 2-column and a 2-row halo. Conv1's four firings bring 10x10, 10x6, 6x10 and 6x6,
    each of the 2 x 16 x 16 elements once, 8192 bits. Where the Add's tile is not copied from Conv1's window, the
    firings also bring per channel 0, 8x2, 2x8 and 8x8 less 6x6 of it: 316 x 2 elements more, 10112 bits."""
    allocation = _solve(tmp_path, SIMBA, _residual(), RESIDUAL)
    space, tr = DecisionSpace(allocation.problem), _into(allocation, "input")
    assert space.moved_bits(tr, allocation.solution.transfer_routes[tr]) * space.iterations == 8192
    assert sorted({space.moved_bits(tr, choice) * space.iterations for choice in space.path_choices[tr]}) == [
        8192,
        10112,
    ]
