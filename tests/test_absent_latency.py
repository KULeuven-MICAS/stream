"""A node absent from a fused loop runs once over the loop's trips: across cores it waits for every trip of the
reduction it reads and then takes its whole latency, which the run's latency includes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, shape_inference

from stream.api import SolveOptions, evaluate_mapping
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.opt.allocation.constraint_optimization.families import overlap
from stream.opt.allocation.constraint_optimization.families.overlap import _drained
from stream.opt.allocation.constraint_optimization.space import DecisionSpace

HARDWARE = Path("stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
MAPPING = """layers:
- name: Gemm1
  core_allocation: [[0]]
- name: Gemm2
  core_allocation: [[1]]
fused_groups:
- name: G
  layers: [Gemm1, Gemm2]
  intra_core_tiling:
"""


K_TILES = "  - {dim: Gemm1.D1, tile: 32}\n"
TOKENS = "  - {dim: Gemm1.D0, tile: 16}\n"
N_TILES = "  - {dim: Gemm2.D2, tile: 32}\n"


def _mlp(path: Path) -> str:
    """X[64,128] -Gemm1(W1[128,256])-> H1[64,256] -Gemm2(W2[256,128])-> Y[64,128], bf16."""
    weights = []
    for name, shape in (("W1", [128, 256]), ("W2", [256, 128])):
        tensor = helper.make_tensor(name, TensorProto.BFLOAT16, shape, np.zeros(shape))
        for field in ("float_data", "int32_data", "raw_data"):
            tensor.ClearField(field)
        weights.append(tensor)
    nodes = [
        helper.make_node("Gemm", ["X", "W1"], ["H1"], name="Gemm1"),
        helper.make_node("Gemm", ["H1", "W2"], ["Y"], name="Gemm2"),
    ]
    values = [
        helper.make_tensor_value_info(n, TensorProto.BFLOAT16, s) for n, s in (("X", [64, 128]), ("Y", [64, 128]))
    ]
    graph = helper.make_graph(nodes, "mlp", values[:1], values[1:], initializer=weights)
    model = shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    onnx.save(model, path / "mlp.onnx")
    return str(path / "mlp.onnx")


def _fast_offchip(path: Path) -> str:
    """The quad-core TPU with the off-chip port and bus at 8192 bits a cycle, so that compute sets the latency."""
    text = HARDWARE.read_text().replace("rw_port_1: {bandwidth: 64,", "rw_port_1: {bandwidth: 8192,")
    text = text.replace("bandwidth: 128 ", "bandwidth: 8192 ")
    (path / "hardware.yaml").write_text(text)
    os.symlink(HARDWARE.resolve().parent / "cores", path / "cores")
    return str(path / "hardware.yaml")


def _solve(tmp_path: Path, tiles: str, gemm2_core: int = 1):
    """The MLP with Gemm1 on core 0 and Gemm2 on ``gemm2_core``, fused along ``tiles`` (innermost first), its cost
    per node and the run's drain."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "mapping.yaml").write_text(MAPPING.replace("[[1]]", f"[[{gemm2_core}]]") + tiles)
    estimate = evaluate_mapping(
        _fast_offchip(tmp_path), _mlp(tmp_path), str(tmp_path), str(tmp_path / "mapping.yaml"), SolveOptions()
    )
    space = DecisionSpace(estimate.context.get("allocation").problem)
    assert all(len(space.path_choices[tr]) == 1 for tr in space.transfer_nodes)
    node = {n.name: n for n in (*space.ssc_nodes, *space.transfer_nodes)}
    whole = {name: space.runtime(n) for name, n in node.items() if n in space.ssc_nodes}
    whole |= {
        name: space.transfer_latency_for_path(n, space.path_choices[n][0])
        for name, n in node.items()
        if n in space.path_choices
    }
    return estimate, whole, max(_drained(space).values())


def test_a_reduction_read_on_another_core_delays_its_reader_by_every_trip(tmp_path: Path):
    """K tiled 32 (4 trips), Gemm1 on core 0, Gemm2 on core 1. Gemm2 reads the whole H1, the sum over every K trip,
    so the run is X and W1's first tiles (the longer), Gemm1's four trips, H1's one firing, Gemm2 whole, Y's one
    firing and the arrays' drain, each at the latency the cost model gives it. The slot breakdown's fill is the
    drain and the absent wait; the iterations take the rest. The solve reports the latency level's objective."""
    estimate, whole, drain = _solve(tmp_path, K_TILES)
    chain = max(whole["Transfer(X)"], whole["Transfer(W1)"]) + 4 * whole["Gemm1"] + whole["Transfer(H1)"]
    assert estimate.cycles == chain + whole["Gemm2"] + whole["Transfer(Y_1)"] + drain
    (group,) = estimate.context.data["group_allocations"].values()
    assert group["solve"]["objective"] == estimate.cycles
    (path,) = tmp_path.rglob("slot_latency_breakdown.yaml")
    totals = yaml.safe_load(path.read_text())["totals"]
    steady = 4 * totals["iteration_latency_cycles"] - 3 * totals["overlap_cycles"]
    assert (totals["fill_cycles"], totals["absent_wait_cycles"]) == (
        estimate.cycles - steady,
        totals["fill_cycles"] - drain,
    )


def test_a_reduction_along_an_outer_loop_completes_at_its_last_trip(tmp_path: Path):
    """K outer (4 trips), tokens inner (tile 16, 4 trips), 16 iterations. Gemm2 is absent from K; its first call reads
    the first token tile of H1, summed over K, complete only after Gemm1's call 3 x 4 + 1 = 13. The run is at least
    W1's first tile, those 13 calls, H1's first firing, Gemm2's four calls, Y's last firing and the drain, as long
    as the run takes."""
    estimate, whole, drain = _solve(tmp_path, "  - {dim: Gemm1.D0, tile: 16}\n  - {dim: Gemm1.D1, tile: 32}\n")
    chain = max(whole["Transfer(X)"], whole["Transfer(W1)"]) + 13 * whole["Gemm1"] + whole["Transfer(H1)"]
    assert estimate.cycles == chain + 4 * whole["Gemm2"] + whole["Transfer(Y_1)"] + drain


@pytest.mark.parametrize(
    ("tiles", "waiting"), [(K_TILES, ["Gemm2"]), (N_TILES, ["Gemm1"]), (TOKENS, [])], ids=["k", "n", "tokens"]
)
def test_a_node_waits_where_it_is_absent_from_a_loop_its_neighbour_varies_on(tmp_path: Path, tiles, waiting):
    """Tiled along K, Gemm2 is absent from the loop Gemm1 reduces along; tiled along Gemm2's outputs, Gemm1 is absent
    from the loop its reader walks; tiled along tokens no node is absent. A waiting node brings the without_wait
    level."""
    ctx = _solve(tmp_path, tiles, 0)[0].context
    space = DecisionSpace(ctx.get("allocation").problem)
    assert sorted(n.name for n in space.waiting_absent_nodes) == waiting
    model = AllocationModel(ctx.get("allocation_problem"), families=ctx.get("families"), backend=ctx.get("backend"))
    assert ("latency_without_wait" in model.objective) == bool(waiting)


def test_the_iterations_report_what_they_need_and_the_wait_the_rest(tmp_path: Path, monkeypatch):
    """The K-tiled MLP on two cores reports the iteration latency and overlap of its steady schedule alone: the wait
    takes all the run adds to them."""
    waited, _, _ = _solve(tmp_path / "waited", K_TILES)
    monkeypatch.setattr(overlap, "_absent_wait", lambda ctx, fill, drained: None)
    steady, _, _ = _solve(tmp_path / "steady", K_TILES)
    keys = ("per_iteration", "overlap_between_iterations")
    assert [_latency(waited)[k] for k in keys] == [_latency(steady)[k] for k in keys]
    assert _latency(waited)["total"] > _latency(steady)["total"]


def _latency(estimate) -> dict[str, int]:
    (group,) = estimate.context.data["group_allocations"].values()
    return group["latency"]
