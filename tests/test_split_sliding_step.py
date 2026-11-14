"""A fused loop over a dim that a split over cores also tiles runs the split inside it: trip k of core c is tile
c + k x cores, so a core's step is its tile times the cores, it keeps from its previous trip only what its window
reaches back past that step, and a neighbour's rows of another trip are sent across cores (bf16, 16 bits)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from conv_windows import HARDWARE, conv, onnx_graph

from stream.api import SolveOptions, evaluate_mapping
from stream.ir import AllocationIR
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.stages.allocation.steady_state_allocation import AllocationStage
from stream.stages.context import StageContext

TPU = HARDWARE.format("tpu_like_quad_core")
CHAIN = """layers:
- {{name: Conv1, core_allocation: [{cores}], inter_core_tiling: [[{{dim: D2, split: {split}}}]]}}
- {{name: Conv2, core_allocation: [{cores}], inter_core_tiling: [[{{dim: D2, split: {split}}}]]}}
fused_groups:
- {{name: g, layers: [Conv1, Conv2], intra_core_tiling: [{{dim: Conv1.D2, tile: {tile}}}]}}
"""


def _solve_chain(rows: int, tile: int, split: int, stride: int) -> Any:
    """The allocation of input (1,4,rows,16) -> Conv1 (8, 3x3, pad 1) -> t1 -> Conv2 (8, 3x3, pad 1, ``stride``),
    both split ``split`` ways along the rows and fused in tiles of ``tile`` rows per core."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out", stride=stride)]
    graph = onnx_graph(nodes, {"input": [1, 4, rows, 16], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])
    with tempfile.TemporaryDirectory() as out:
        path = Path(out) / "mapping.yaml"
        path.write_text(CHAIN.format(cores=list(range(split)), split=split, tile=tile))
        return evaluate_mapping(TPU, graph, out, str(path), SolveOptions(artifacts=False)).context.get("allocation")


def _transfers(tile: int) -> tuple[dict[str, dict[str, Any]], int]:
    """The IR of the transfers by name of the 12-row chain split two ways, and the iterations."""
    steady = AllocationIR.from_internal(_solve_chain(12, tile, 2, 1)).model_dump()["steady_state"]
    transfers = {n["name"]: n for n in steady["tiled_graph"]["nodes"] if n["kind"] == "transfer"}
    return transfers, steady["iterations"]


def _crossing_bits(transfer: dict[str, Any], iterations: int) -> float:
    """Bits of the transfer sent between different cores over the run."""
    per_firing = sum(h["elements"] for h in transfer["demand"] if h["source"] != h["target"])
    return per_firing * 16 * iterations / transfer["reuse_factor"]


def test_a_cores_steps_that_do_not_touch_keep_nothing_and_hand_the_neighbours_rows():
    """Rows by 3: 2 trips of 6 rows, core 0 makes rows 6k..6k+2, core 1 6k+3..6k+5. Conv2's window on core 0 is rows
    0..3 at trip 0 and 5..9 at trip 1: they do not meet, so the temporal loop keeps no halo. Rows 3, 2, 6, 5, 9 and 8
    cross cores: 6 rows of 8 channels x 16 columns, 12288 bits."""
    transfers, iterations = _transfers(3)
    transfer = transfers["Transfer(t1)"]
    assert {h["type"]: h["halo"] for h in transfer["halos"]} == {"spatial": 2}
    assert _crossing_bits(transfer, iterations) == 6 * 8 * 16 * 16


def test_a_window_reaching_past_a_cores_step_keeps_the_rest_and_takes_new_rows_from_the_neighbour():
    """Rows by 1: 6 trips of 2 rows, core c makes row 2k + c and Conv2 there reads rows 2k+c-1..2k+c+1, keeping 1 row
    of its previous window. Core 0 takes row 2k+1 from core 1, core 1 row 2k+2 from core 0 (and row 0 at trip 0, none
    at trip 5): 12 rows of 8 x 16 cross cores, 24576 bits. Off chip sends Conv1 the union of what both cores lack,
    rows 2k..2k+2 at trips 0 to 4 and rows 10..11 at trip 5: 17 rows of 4 x 16, 17408 bits."""
    transfers, iterations = _transfers(1)
    transfer = transfers["Transfer(t1)"]
    assert {h["type"]: h["halo"] for h in transfer["halos"]} == {"temporal": 1, "spatial": 2}
    assert _crossing_bits(transfer, iterations) == 12 * 8 * 16 * 16
    moved = transfers["Transfer(input)"]
    assert moved["moved_bits"] * iterations / moved["reuse_factor"] == pytest.approx(17 * 4 * 16 * 16)


@pytest.mark.parametrize(("rows", "tile", "stride", "lead"), [(12, 1, 1, 1), (16, 2, 2, 0)], ids=["ahead", "behind"])
def test_a_reader_of_a_neighbours_next_trip_waits_one_producer_firing(monkeypatch, rows, tile, stride, lead):
    """Two ways, rows by 1: Conv2 on core 1 reads row 2k + 2 at trip k, which core 0 makes one firing later: the fill
    holds one Conv1 firing more than without the lead. Stride 2, rows by 2: Conv2 on core c reads rows
    4k + 2c - 1..4k + 2c + 1, none of a later trip, so nothing waits."""
    allocation = _solve_chain(rows, tile, 2, stride)
    conv1 = next(n for n in allocation.space.ssc_nodes if n.name == "Conv1")
    assert allocation.space.neighbour_waits == ({conv1: lead} if lead else {})
    assert allocation.space.warmup.get(conv1, 0) == lead
    fill = AllocationIR.from_internal(allocation).model_dump()["latency"]["fill"]
    monkeypatch.setattr(DecisionSpace, "neighbour_waits", property(lambda _: {}))
    without = AllocationIR.from_internal(_solve_chain(rows, tile, 2, stride)).model_dump()["latency"]["fill"]
    assert fill - without == lead * allocation.space.active_runtime(conv1)


def test_aie_code_generation_refuses_a_fused_loop_over_a_split_sliding_dim_before_the_solve():
    """Rows by 1 two ways sends a core its neighbour's rows of other trips, which no AIE lowering emits: code
    generation refuses the group before it is solved. One core fused by rows passes."""
    aie_code_generation = pytest.importorskip(
        "stream.stages.codegen.aie_code_generation",
        reason="the AIE dialects are a separate install, via stream-setup-aie",
    )
    problem = _solve_chain(12, 1, 2, 1).problem
    ctx = StageContext.from_kwargs(accelerator=problem.accelerator, allocation_problem=problem)
    stage = aie_code_generation.AIECodeGenerationStage([AllocationStage], ctx)
    with pytest.raises(ValueError, match=r"slides a window past a core's tile along \['z\d+'\], which a fused loop"):
        next(stage.run())
    aie_code_generation.reject_fused_split_windows(_solve_chain(12, 3, 1, 1).problem)
