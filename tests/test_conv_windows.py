"""Fused convolutions tiled along a sliding window: the oracle against the hand-derived sizes, and every solved
allocation against the oracle (bf16 elements; per core in the mapping's core order; interior iteration)."""

import tempfile
from collections import Counter

import pytest
from conv_windows import (
    HARDWARE,
    MAPPINGS,
    SCENARIOS,
    Scenario,
    _block,
    _box,
    _shape,
    conv,
    conv_chain,
    onnx_graph,
    oracle,
    reads,
    solve,
)
from onnx import helper

from stream.api import SolveOptions, evaluate_mapping
from stream.frontends import load_workload
from stream.opt.allocation.constraint_optimization.families import overlap
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator

C1, C2 = (1, 16, 32, 32), (1, 32, 32, 32)
SPEC = {
    "s1": {
        "iterations": 1,
        "conv1_tile": (C1,),
        "out_tile": (C2,),
        "input_staged": ((1, 8, 32, 32),),
        "input_moved": ((1, 8, 32, 32),),
        "window": (C1,),
        "moved": (C1,),
        "sources": {0: {0: 16384}},
        "conv1_rows": (32,),
        "input_moved_rows": (32,),
    },
    "s2": {
        "iterations": 4,
        "conv1_tile": ((1, 16, 8, 32),),
        "out_tile": ((1, 32, 8, 32),),
        "input_staged": ((1, 8, 10, 32),),
        "input_moved": ((1, 8, 8, 32),),
        "window": ((1, 16, 10, 32),),
        "moved": ((1, 16, 8, 32),),
        "sources": {0: {0: 4096}},
        "conv1_rows": (9, 8, 8, 7),
        "input_moved_rows": (10, 8, 8, 6),
    },
    "s3": {
        "iterations": 1,
        "conv1_tile": ((1, 16, 32, 8),) * 4,
        "out_tile": ((1, 32, 32, 8),) * 4,
        "input_staged": ((1, 8, 32, 9), (1, 8, 32, 10), (1, 8, 32, 10), (1, 8, 32, 9)),
        "input_moved": ((1, 8, 32, 9), (1, 8, 32, 10), (1, 8, 32, 10), (1, 8, 32, 9)),
        "window": ((1, 16, 32, 9), (1, 16, 32, 10), (1, 16, 32, 10), (1, 16, 32, 9)),
        "moved": ((1, 16, 32, 9), (1, 16, 32, 10), (1, 16, 32, 10), (1, 16, 32, 9)),
        "sources": {
            0: {0: 4096, 1: 512},
            1: {0: 512, 1: 4096, 2: 512},
            2: {1: 512, 2: 4096, 3: 512},
            3: {2: 512, 3: 4096},
        },
        "conv1_rows": (32,),
        "input_moved_rows": (32,),
    },
    "s4": {
        "iterations": 4,
        "conv1_tile": ((1, 16, 8, 16),) * 2,
        "out_tile": ((1, 32, 8, 16),) * 2,
        "input_staged": ((1, 8, 10, 17),) * 2,
        "input_moved": ((1, 8, 8, 17),) * 2,
        "window": ((1, 16, 10, 17),) * 2,
        "moved": ((1, 16, 8, 17),) * 2,
        "sources": {2: {0: 2048, 1: 128}, 3: {0: 128, 1: 2048}},
        "conv1_rows": (9, 8, 8, 7),
        "input_moved_rows": (10, 8, 8, 6),
    },
    "s5": {
        "iterations": 4,
        "conv1_tile": ((1, 16, 8, 32),),
        "out_tile": ((1, 32, 4, 16),),
        "input_staged": ((1, 8, 10, 32),),
        "input_moved": ((1, 8, 8, 32),),
        "window": ((1, 16, 9, 32),),
        "moved": ((1, 16, 8, 32),),
        "sources": {0: {0: 4096}},
        "conv1_rows": (8, 8, 8, 8),
        "input_moved_rows": (9, 8, 8, 7),
    },
}
SPEC["pool"] = SPEC["s4"] | {"out_tile": ((1, 16, 8, 16),) * 2}
SPEC["s5_named"] = SPEC["s5"]
SPEC["valid"] = {
    "iterations": 1,
    "conv1_tile": ((1, 16, 30, 30),),
    "out_tile": ((1, 32, 28, 28),),
    "input_staged": ((1, 8, 32, 32),),
    "input_moved": ((1, 8, 32, 32),),
    "window": ((1, 16, 30, 30),),
    "moved": ((1, 16, 30, 30),),
    "sources": {0: {0: 14400}},
    "conv1_rows": (30,),
    "input_moved_rows": (32,),
}
SPEC["s6"] = SPEC["s3"] | {
    "out_tile": ((1, 32, 16, 4),) * 4,
    "window": ((1, 16, 32, 8), (1, 16, 32, 9), (1, 16, 32, 9), (1, 16, 32, 9)),
    "moved": ((1, 16, 32, 8), (1, 16, 32, 9), (1, 16, 32, 9), (1, 16, 32, 9)),
    "sources": {0: {0: 4096}, 1: {0: 512, 1: 4096}, 2: {1: 512, 2: 4096}, 3: {2: 512, 3: 4096}},
}
CHECKS = tuple(SPEC["s1"])
IN_PLACE = {"s3": False, "s4": False, "pool": False, "s6": False}
QUAD = ("s1", "s2", "s3", "s4", "s5", "pool", "valid", "s5_named", "s6")
RUNS = [
    *((s, hw) for s in QUAD for hw in ("eyeriss_like_quad_core", "tpu_like_quad_core")),
    *((s, "simba_small") for s in ("s1", "s2", "s3", "s4", "s5", "pool", "s6")),
    *((s, "fusemax") for s in ("s1", "s2", "s5", "s3_shared", "s4_shared")),
]
FUSED = [run for run in RUNS if SCENARIOS[run[0]].rows and run[1] != "simba_small"]


def _run(scenario: str, hardware: str):
    return pytest.param(scenario, hardware, marks=[pytest.mark.slow] if hardware == "simba_small" else [])


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("scenario", SPEC)
def test_oracle_matches_the_hand_derived_sizes(scenario: str, check: str):
    assert oracle(scenario)[check] == SPEC[scenario][check]


@pytest.mark.parametrize(("scenario", "hardware"), [_run(*run) for run in RUNS])
def test_allocation_matches_the_oracle(scenario: str, hardware: str):
    """Every tile is the oracle's; conv2's window keeps the halo consecutive tiles share; what other cores hand over
    moves over the links on one DMA channel per core; the first tiles are longer by the lookahead and the halo, and
    conv2 waits for conv1's where it runs on other cores."""
    found, expected = solve(scenario, hardware), oracle(scenario)
    assert {check: found[check] for check in CHECKS} == {check: expected[check] for check in CHECKS}
    windows, moved, cores = expected["window"], expected["moved"], len(expected["window"])
    beyond = sum(window[3] for window in windows) - sum(tile[3] for tile in expected["conv1_tile"])
    halos = {"TEMPORAL": windows[0][2] - moved[0][2], "SPATIAL": beyond // (cores - 1) if cores > 1 else 0}
    assert found["halos"] == {loop: halo for loop, halo in halos.items() if halo}
    assert found["in_place"] is IN_PLACE.get(scenario, True)
    if not found["in_place"]:
        handed = {(src, dst): n for dst, srcs in expected["sources"].items() for src, n in srcs.items() if src != dst}
        assert found["linked_bits"] == 16 * sum(handed.values())
        fan_in, fan_out = (max(Counter(pair[k] for pair in handed).values()) for k in (1, 0))
        assert found["fan"] == (fan_in, fan_out)
        assert found["target_shares"] == [1.0, 1.0]
    rows, inputs = expected["conv1_rows"], expected["input_moved_rows"]
    first = {"Conv1": rows[0] * len(rows) / sum(rows) - 1, "Transfer(input)": inputs[0] * len(inputs) / sum(inputs) - 1}
    assert {name: found["first_tiles"].get(name, 0.0) for name in first} == first
    conv1_cores, conv2_cores = (
        set(cores) for cores in (SCENARIOS[scenario].conv1_cores, SCENARIOS[scenario].conv2_cores)
    )
    assert found["waits_elsewhere"] is not conv1_cores.issuperset(conv2_cores)


@pytest.mark.parametrize(
    ("stride", "sizes"), [(1, [1, 3, 3, 3, 3, 8, 16, 32, 32, 32]), (2, [1, 2, 2, 3, 3, 3, 3, 8, 16, 16, 16, 32])]
)
def test_fused_axes_are_one_unique_dimension(stride: int, sizes: list[int]):
    workload = load_workload(conv_chain(stride))
    assert sorted(workload.get_dimension_size(z) for z in workload.unique_dimensions()[0]) == sizes


@pytest.mark.parametrize("tile", [None, 0, 3])
def test_a_reader_footprint_is_its_window_and_the_producer_footprint_its_tile(tile: int | None):
    """An 8-row OY tile: conv2 reads the rows the oracle enumerates for that tile, conv1 writes 8 rows."""
    workload = load_workload(conv_chain(1))
    conv1, conv2 = workload.get_computation_nodes()
    sizes = {z: workload.get_dimension_size(z) for z in workload.unique_dimensions()[0]}
    sizes[oy := workload.get_dims(conv2)[2]] = 8
    at = None if tile is None else {oy: tile}
    mid, start = conv2.inputs[0], 8 * (1 if tile is None else tile)
    tile_out = {(0, k, y, x) for k in range(32) for y in range(start, start + 8) for x in range(32)}
    rows = len({index[2] for index in reads(conv2, mid, tile_out)})
    assert workload.get_tensor_shape_with_dimension_sizes(mid, sizes, conv2, at) == (1, 16, rows, 32)
    assert workload.get_tensor_shape_with_dimension_sizes(mid, sizes, at=at) == (1, 16, 8, 32)
    assert workload.get_tensor_shape_with_dimension_sizes(mid, sizes, conv1, at) == (1, 16, 8, 32)


def _tiled(rows: int):
    workload = load_workload(conv_chain(1))
    sizes = {z: workload.get_dimension_size(z) for z in workload.unique_dimensions()[0]}
    sizes[workload.get_dims(workload.get_computation_nodes()[1])[2]] = rows
    return workload.with_modified_dimension_sizes(sizes)


def test_zigzag_costs_the_interior_tile_without_border_padding():
    """Tiled to 8 output rows, conv2 reads a 10-row window of conv1_out, of which it pads nothing."""
    tiled = _tiled(8)
    estimator = ZigZagCostEstimator(workload=tiled, accelerator=None, mapping=None)  # type: ignore[arg-type]
    _, _, pr_sizes = estimator.create_equation_and_dimension_relations_and_pr_sizes(tiled.get_computation_nodes()[1])
    assert sorted(pr_sizes.data.values()) == [10, 32]


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_conv1_computes_what_the_window_first_reaches_however_far_ahead(rows: int):
    """A one-row tile reaches as far ahead as it steps, so conv1 is done a tile early: its last tile is empty."""
    tiled = _tiled(rows)
    conv1, conv2 = tiled.get_computation_nodes()
    work = tiled.get_sliding_work(tiled.get_dims(conv2)[2], 32 // rows)
    assert work == {conv1: oracle(Scenario(1, rows, (0,), (0,)))["conv1_rows"], conv2: (rows,) * (32 // rows)}


@pytest.mark.slow
@pytest.mark.parametrize(("scenario", "hardware"), FUSED)
def test_the_first_tiles_lengthen_the_fill(scenario: str, hardware: str, monkeypatch):
    fill = solve(scenario, hardware)["fill"]
    monkeypatch.setattr(overlap, "_warmup", lambda _ctx: None)
    assert fill > solve.__wrapped__(scenario, hardware)["fill"]


@pytest.mark.parametrize(
    ("graph", "node", "extents"),
    [
        ("valid", "Conv1", [30, 30]),
        ("valid_beside_same", "Conv2", [32, 32]),
    ],
)
def test_a_producer_keeps_its_own_extent_under_an_unpadded_reader(graph: str, node: str, extents: list[int]):
    """An unpadded 3x3 reader of a 30-row tensor spans 28 rows: the producer's axis stays its own, sized by itself."""
    weights = {"w1": [16, 8, 3, 3], "w2": [32, 16, 3, 3], "w3": [32, 16, 3, 3]}
    nodes = {
        "valid": [conv("Conv1", "input", "w1", "m", pads=0), conv("Conv2", "m", "w2", "y", pads=0)],
        "valid_beside_same": [
            conv("Conv1", "input", "w1", "m"),
            conv("Conv2", "m", "w2", "y"),
            conv("Conv3", "m", "w3", "z", pads=0),
        ],
    }[graph]
    workload = load_workload(onnx_graph(nodes, {"input": [1, 8, 32, 32]} | weights, ["y", "z"][: len(nodes) - 1]))
    found = workload.get_node_by_name(node)
    assert [workload.get_dimension_size(d) for d in workload.get_dims(found)][1:3] == extents


def test_a_transfer_to_two_windowed_readers_hands_each_core_their_widest_window():
    """conv1_out feeds a 3x3 and a 5x5 conv split over the same columns: the 5x5's two halo columns are handed."""
    shapes = {"input": [1, 8, 32, 32], "w1": [16, 8, 3, 3], "w2": [32, 16, 3, 3], "w3": [32, 16, 5, 5]}
    nodes = [
        conv("Conv1", "input", "w1", "m"),
        conv("Conv2", "m", "w2", "y"),
        conv("Conv3", "m", "w3", "z", k=5, pads=2),
        helper.make_node("Add", ["y", "z"], ["o"], name="Add"),
    ]
    path = onnx_graph(nodes, shapes, ["o"])
    with tempfile.TemporaryDirectory() as out:
        estimate = evaluate_mapping(
            HARDWARE.format("eyeriss_like_quad_core"),
            path,
            out,
            str(MAPPINGS / "branch.yaml"),
            SolveOptions(artifacts=False),
        )
    space = DecisionSpace(estimate.context.get("allocation").problem)
    transfer = next(tr for tr in space.transfer_nodes if tr.name == "Transfer(m)")
    conv3 = load_workload(path).get_node_by_name("Conv3")
    owned = [_box((1, 16, 32, 32), range(32), _block(32, 4, i)) for i in range(4)]
    read = [reads(conv3, conv3.inputs[0], _box((1, 32, 32, 32), range(32), _block(32, 4, j))) for j in range(4)]
    expected = {(i, j): n for i in range(4) for j in range(4) if (n := len(owned[i] & read[j]))}
    assert space.overlaps(transfer) == expected


def test_a_core_position_counts_the_last_split_dim_fastest_as_codegen_unrolls_cores():
    """``iterate_spat_vars`` gives core c the tile (c // 4, c % 4) of an (OY 2, OX 4) split: edge columns hold 9."""
    workload = load_workload(conv_chain(1))
    conv2 = workload.get_computation_nodes()[1]
    tiling = ((workload.get_dims(conv2)[2], 2), (workload.get_dims(conv2)[1], 4))
    found = [workload.get_tensor_shape_with_tiling(conv2.inputs[0], tiling, conv2, c) for c in range(8)]
    tiles = [_box((1, 32, 32, 32), _block(32, 2, c // 4), _block(32, 4, c % 4)) for c in range(8)]
    assert found == [_shape(reads(conv2, conv2.inputs[0], tile)) for tile in tiles]


def test_a_strided_residual_block_solves():
    """A stride-2 3x3 and a stride-2 1x1 downsample read the same input; their sum fuses into one group."""
    shapes = {"input": [1, 8, 32, 32], "w1": [16, 8, 3, 3], "w2": [16, 16, 3, 3], "wd": [16, 8, 1, 1]}
    nodes = [
        conv("Conv1", "input", "w1", "a", stride=2),
        conv("Conv2", "a", "w2", "b"),
        conv("Down", "input", "wd", "d", k=1, stride=2, pads=0),
        helper.make_node("Add", ["b", "d"], ["o"], name="Add"),
    ]
    with tempfile.TemporaryDirectory() as out:
        estimate = evaluate_mapping(HARDWARE.format("tpu_like_quad_core"), onnx_graph(nodes, shapes, ["o"]), out)
    assert estimate.cycles > 0
