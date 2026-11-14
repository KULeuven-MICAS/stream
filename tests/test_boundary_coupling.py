"""A windowed reader whose extent differs from its producer's at the far edge (an unpadded window, a stride that does
not divide the producer's extent) fuses with it: the producer's last tile, in time or across cores, makes the edge."""

from __future__ import annotations

import logging
from math import prod
from pathlib import Path

import pytest
from conv_windows import HARDWARE, conv, onnx_graph
from zigzag.datatypes import LayerDim as ZigZagLayerDim
from zigzag.utils import open_yaml

from stream.allocation.allocation import Allocation
from stream.api import SolveOptions, evaluate_mapping
from stream.frontends import load_workload
from stream.mapping.work_share import interchangeable
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_parser import MappingParser
from stream.stages.context import StageContext
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.stage import LeafStage
from stream.workload.utils import determine_fusion_splits
from stream.workload.workload import Workload

EYERISS = HARDWARE.format("eyeriss_like_single_core")
SIMBA = HARDWARE.format("simba_small")
FUSED = """
layers:
- {{name: Conv1, core_allocation: [[{cores}]]{split}}}
- {{name: Conv2, core_allocation: [[{cores}]]{split}}}
fused_groups:
- {{name: g, layers: [Conv1, Conv2], intra_core_tiling: [{tiles}]}}
"""
ROWS_SPLIT = ", inter_core_tiling: [[{dim: D2, split: 2}]]"
"""Each conv's output rows (its dim D2; D1 is its columns) split over 2 cores."""


def _valid() -> str:
    """input (1,4,14,18) -> Conv1 (8, 3x3, pad 0) -> t1 (1,8,12,16) -> Conv2 (8, 3x3, pad 0) -> (1,8,10,14)."""
    nodes = [conv("Conv1", "input", "w1", "t1", pads=0), conv("Conv2", "t1", "w2", "out", pads=0)]
    return onnx_graph(nodes, {"input": [1, 4, 14, 18], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


def _strided() -> str:
    """input (1,4,11,15) -> Conv1 (8, 3x3, pad 1) -> t1 (1,8,11,15) -> Conv2 (8, 3x3, stride 2, pad 1) -> (1,8,6,8)."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out", stride=2)]
    return onnx_graph(nodes, {"input": [1, 4, 11, 15], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


def _refused() -> str:
    """input (1,4,12,12) -> Conv1 (8, 3x3, pad 1) -> t1 (1,8,12,12) -> Conv2 (8, 1x1, pad 1) -> (1,8,14,14)."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out", k=1)]
    return onnx_graph(nodes, {"input": [1, 4, 12, 12], "w1": [8, 4, 3, 3], "w2": [8, 8, 1, 1]}, ["out"])


def _mapping(tiles: str, cores: str = "0", split: str = "") -> str:
    return FUSED.format(tiles=tiles, cores=cores, split=split)


def _solve(tmp_path: Path, hardware: str, workload: str, mapping: str) -> Allocation:
    (tmp_path / "mapping.yaml").write_text(mapping)
    options = SolveOptions(artifacts=False)
    estimate = evaluate_mapping(hardware, workload, str(tmp_path), str(tmp_path / "mapping.yaml"), options)
    return estimate.context.get("allocation")


def _run_bits(allocation: Allocation, tensor: str) -> float:
    """Bits the transfer of ``tensor`` moves over the run in the solved routes: per firing, once per its reuse."""
    space, solution = DecisionSpace(allocation.problem), allocation.solution
    tr = next(tr for tr in allocation.problem.workload.get_transfer_nodes() if tr.inputs[0].name == tensor)
    return space.moved_bits(tr, solution.transfer_routes[tr]) * space.iterations / solution.reuse_factors[tr]


def _held(allocation: Allocation, name: str) -> int:
    """Bits the solved allocation holds of the tensor ``name`` over all memories."""
    rows = allocation.solution.performance["memory_occupancy"]
    return sum(entry["bits"] for row in rows for entry in row["tensors"] if entry["tensor"] == name)


def _work(allocation: Allocation, dim_of: str, axis: int) -> dict[str, tuple[int, ...]]:
    """Each node's work along the unique dim ``dim_of``'s output ``axis`` (1 columns, 2 rows) slides, by name."""
    workload, splits = allocation.problem.workload, allocation.problem.fusion_splits
    dim = workload.get_dims(workload.get_node_by_name(dim_of))[axis]
    return {node.name: work for node, work in workload.get_sliding_work(dim, splits[dim]).items()}


def _axes(workload: Workload, name: str) -> list:
    return workload.get_dims(workload.get_node_by_name(name))[1:3]


def test_an_unpadded_reader_shares_its_producers_axes_sized_by_itself():
    """Conv2's 14 columns read t1 columns j..j+2, 16 in all: s*(O-1)+1 = 14 <= 16 <= s*O + 3 - 1 = 16, and its 10
    rows read t1's 12 alike, so Conv1's columns and rows are Conv2's, sized by Conv2."""
    workload = load_workload(_valid())
    assert _axes(workload, "Conv1") == _axes(workload, "Conv2")
    assert [workload.get_dimension_size(d) for d in _axes(workload, "Conv2")] == [14, 10]


def test_a_stride_that_does_not_divide_its_producer_shares_its_axes():
    """Conv2 (stride 2, pad 1) makes 8 columns from t1's 15 and 6 rows from its 11: 2*7+1 = 15 <= 15 <= 2*8 + 3 - 1
    and 2*5+1 = 11 <= 11, so Conv1's columns and rows step twice as fast along Conv2's."""
    workload = load_workload(_strided())
    ox, oy = _axes(workload, "Conv2")
    assert [workload.leading_dim(d) for d in _axes(workload, "Conv1")] == [(ox, 2), (oy, 2)]


def test_a_reader_reaching_past_the_window_leaves_the_producer_uncoupled_with_a_warning(caplog):
    """A 1x1 reader padded by 1 makes 14 rows from Conv1's 12: s*(O-1)+1 = 14 > 12, so no tile of it maps onto
    Conv1's rows. Conv1 keeps its rows and columns its own, of 12, and the refusal names both nodes."""
    with caplog.at_level(logging.WARNING, logger="stream.workload.workload"):
        workload = load_workload(_refused())
        conv1 = _axes(workload, "Conv1")
    assert conv1 != _axes(workload, "Conv2")
    assert [workload.get_dimension_size(d) for d in conv1] == [12, 12]
    assert "Conv2 reads an axis of Conv1 of extent 12 for 14 outputs at stride 1" in caplog.text


@pytest.mark.slow
def test_an_unpadded_producer_makes_its_edge_in_the_last_tile(tmp_path):
    """Conv2's tile 0 reads t1 columns 0..8, tile 1 7..15: Conv1 makes 9 then 7 columns of its own 12 rows, and the
    input moves 11 then 7 columns, each of its 4 x 14 x 18 elements once, 16128 bits; t1's tile, double-buffered,
    holds 2 x 8 x 12 x 7 elements, 21504 bits."""
    allocation = _solve(tmp_path, EYERISS, _valid(), _mapping("{dim: Conv2.D1, tile: 7}"))
    work = _work(allocation, "Conv2", 1)
    assert (work["Conv1"], work["Conv2"], work["Transfer(input)"]) == ((9, 7), (7, 7), (11, 7))
    assert _run_bits(allocation, "input") == 4 * 14 * 18 * 16
    assert _held(allocation, "t1") == 2 * 8 * 12 * 7 * 16


@pytest.mark.slow
def test_a_stride_that_does_not_divide_its_producer_clips_the_last_tile(tmp_path):
    """Conv2's tile k of 2 columns reads t1 columns 4k-1..4k+3: Conv1 makes 4 a tile, the last clipped at t1's end,
    columns 12..14 of 15, and the input moves 5, 4, 4, then the 2 left: each of its 4 x 11 x 15 elements once, 10560
    bits. Conv1's double-buffered tile spans its own 11 rows, not 12."""
    allocation = _solve(tmp_path, EYERISS, _strided(), _mapping("{dim: Conv2.D1, tile: 2}"))
    work = _work(allocation, "Conv2", 1)
    assert (work["Conv1"], work["Transfer(input)"]) == ((4, 4, 4, 3), (5, 4, 4, 2))
    assert _run_bits(allocation, "input") == 4 * 11 * 15 * 16
    assert _held(allocation, "t1") == 2 * 8 * 11 * 4 * 16


@pytest.mark.slow
def test_a_strided_reader_of_an_odd_input_moves_its_rows_once(tmp_path):
    """input (1,8,17,17) -> Conv1 (16, 3x3, stride 2, pad 1) -> (1,16,9,9) -> Conv2 (1x1), Conv1's rows in tiles of 3:
    output row r reads input rows 2r-1..2r+1, so the input moves 6, 6 and 5 new rows, 17 in all, each of its
    8 x 17 x 17 elements once, 36992 bits, not 18 rows' 39168."""
    nodes = [conv("Conv1", "input", "w1", "t1", stride=2), conv("Conv2", "t1", "w2", "out", k=1, pads=0)]
    workload = onnx_graph(nodes, {"input": [1, 8, 17, 17], "w1": [16, 8, 3, 3], "w2": [16, 16, 1, 1]}, ["out"])
    mapping = _mapping("{dim: Conv1.D6, tile: 8}, {dim: Conv1.D2, tile: 3}")
    allocation = _solve(tmp_path, SIMBA, workload, mapping)
    assert _work(allocation, "Conv1", 2)["Transfer(input)"] == (6, 6, 5)
    assert _run_bits(allocation, "input") == 8 * 17 * 17 * 16


@pytest.mark.slow
def test_the_last_core_of_a_split_makes_the_producers_edge(tmp_path):
    """The unpadded chain with rows split over 2 cores: Conv2's 10 rows as 0..4 and 5..9 read t1 rows 0..6 and 5..11.
    Conv1's first core makes rows 0..4, its last 5..11, the edge Conv2's extent leaves out included; their windows of
    the input are rows 0..6 and 5..13, all 14 between them, each of the 4 x 14 x 18 elements moving once, 16128 bits."""
    allocation = _solve(tmp_path, SIMBA, _valid(), _mapping("{dim: Conv2.D1, tile: 7}", "0, 1", ROWS_SPLIT))
    workload, mapping = allocation.problem.workload, allocation.problem.mapping
    conv1 = workload.get_node_by_name("Conv1")
    rows = [workload.get_tensor_single_core(conv1.outputs[0], conv1, mapping, core).shape[2] for core in (0, 1)]
    assert rows == [5, 7]
    assert _run_bits(allocation, "input") == 4 * 14 * 18 * 16


@pytest.fixture(scope="module")
def tiled(tmp_path_factory: pytest.TempPathFactory):
    """``(workload, mapping, hardware)`` -> the tiling stage over them, its tiled workload, mapping and the costing."""

    def tile(workload: str, mapping: str, hardware: str = EYERISS):
        validator = AcceleratorValidator(open_yaml(hardware), hardware)
        assert validator.validate()
        accelerator = AcceleratorFactory(validator.normalized_data).create()
        loaded, path = load_workload(workload), tmp_path_factory.mktemp("tiled") / "mapping.yaml"
        path.write_text(mapping)
        parsed = MappingParser(str(path), loaded, accelerator).run()
        stage = TilingGenerationStage(
            [LeafStage], StageContext.from_kwargs(workload=loaded, mapping=parsed, output_path="")
        )
        stage.fusion_splits = determine_fusion_splits(loaded, parsed)
        small = loaded.with_modified_dimension_sizes(stage.substitute_loop_sizes_with_tiled_sizes())
        estimator = ZigZagCostEstimator(small, accelerator, parsed.with_updated_workload(small, loaded))
        estimator.fusion_splits = stage.fusion_splits
        return stage, estimator

    return tile


def _blocks(estimator: ZigZagCostEstimator, name: str) -> list[dict[tuple[int, int], float]]:
    """Per core of node ``name``, each (rows, columns) it makes a tile at with the share of its tiles at it."""
    node = estimator.workload.get_node_by_name(name)
    found = []
    for core in estimator.mapping.get(node).resource_allocation[0]:
        shares: dict[tuple[int, int], float] = {}
        for sizes, share in estimator.blocks(node, core):
            key = (sizes[ZigZagLayerDim("D2")], sizes[ZigZagLayerDim("D1")])
            shares[key] = shares.get(key, 0.0) + share
        found.append(shares)
    return found


def _costed_macs(estimator: ZigZagCostEstimator, name: str) -> float:
    """The MACs node ``name``'s costed tiles add up to over the run: per core, its tiles' average, times the tiles."""
    node = estimator.workload.get_node_by_name(name)
    tiles = prod(estimator.fusion_splits.values())
    cores = estimator.mapping.get(node).resource_allocation[0]
    return tiles * sum(prod(s.data.values()) * share for core in cores for s, share in estimator.blocks(node, core))


def test_the_group_counts_a_coupled_producers_macs_at_its_own_extent(tiled):
    """The unpadded chain: Conv1 makes 8 x 12 x 16 outputs of 4 x 3 x 3 MACs, 55296, though Conv2's axes are 10 x 14;
    Conv2 makes 8 x 10 x 14 of 8 x 3 x 3, 80640; 135936 in all."""
    stage, _ = tiled(_valid(), _mapping("{dim: Conv2.D1, tile: 7}"))
    assert stage._total_mac_ops() == 8 * 12 * 16 * 4 * 9 + 8 * 10 * 14 * 8 * 9


@pytest.mark.parametrize(
    ("workload", "tile", "split", "conv1", "macs"),
    [
        (_valid, "{dim: Conv2.D1, tile: 7}", "", [{(12, 7): 0.5, (12, 9): 0.5}], 8 * 12 * 16 * 4 * 9),
        (_strided, "{dim: Conv2.D1, tile: 2}", "", [{(11, 4): 0.75, (11, 3): 0.25}], 8 * 11 * 15 * 4 * 9),
        (
            _valid,
            "{dim: Conv2.D1, tile: 7}",
            ROWS_SPLIT,
            [{(5, 7): 0.5, (5, 9): 0.5}, {(7, 7): 0.5, (7, 9): 0.5}],
            8 * 12 * 16 * 4 * 9,
        ),
    ],
)
def test_zigzag_costs_a_coupled_producer_at_its_own_extent_with_the_edge_in_its_last_tile(
    tiled, workload, tile, split, conv1, macs
):
    """Conv1's (rows, columns) per tile with their shares: the unpadded chain's 12 rows and 7 then 9 of its 16 columns;
    the stride-2 chain's 11 rows and 4, 4, 4 then 3 of its 15 columns; split over 2 cores, the first makes 5 rows and
    the last 7. Weighted by share, the costed MACs over the run are Conv1's own."""
    cores = "0, 1" if split else "0"
    _, estimator = tiled(workload(), _mapping(tile, cores, split), SIMBA if split else EYERISS)
    blocks = _blocks(estimator, "Conv1")
    assert [{k: pytest.approx(v) for k, v in core.items()} for core in conv1] == blocks
    assert _costed_macs(estimator, "Conv1") == pytest.approx(macs)


def test_cores_making_different_edges_are_costed_apart(tiled):
    """Split over 2 cores, Conv1's last core makes 7 rows to its first's 5, so neither's cost stands for the other's;
    Conv2's two cores make 5 rows each and share one."""
    _, estimator = tiled(_valid(), _mapping("{dim: Conv2.D1, tile: 7}", "0, 1", ROWS_SPLIT), SIMBA)
    workload, mapping, splits = estimator.workload, estimator.mapping, estimator.fusion_splits
    same = []
    for name in ("Conv1", "Conv2"):
        node = workload.get_node_by_name(name)
        one, other = mapping.get(node).resource_allocation[0]
        same.append(interchangeable(workload, mapping, node, one, other, splits))
    assert same == [False, True]


def test_producers_of_equal_tiles_and_different_edges_are_not_equal(tiled):
    """Three unpadded 8-channel convs over 18 x 18 in tiles of Conv3's 6 x 6 of 12 x 12: Conv2 and Conv3 make 6 x 6
    tiles from 8 x 8 windows alike, but Conv2's buffer is 14 x 14 and Conv3's 12 x 12: only Conv2 reaches an edge."""
    nodes = [
        conv(f"Conv{i}", src, f"w{i}", dst, pads=0)
        for i, (src, dst) in enumerate([("input", "t1"), ("t1", "t2"), ("t2", "out")], start=1)
    ]
    shapes = {"input": [1, 8, 18, 18], "w1": [8, 8, 3, 3], "w2": [8, 8, 3, 3], "w3": [8, 8, 3, 3]}
    mapping = """
layers:
- {name: Conv1, core_allocation: [[0]]}
- {name: Conv2, core_allocation: [[0]]}
- {name: Conv3, core_allocation: [[0]]}
fused_groups:
- {name: g, layers: [Conv1, Conv2, Conv3], intra_core_tiling: [{dim: Conv3.D1, tile: 6}, {dim: Conv3.D2, tile: 6}]}
"""
    _, estimator = tiled(onnx_graph(nodes, shapes, ["out"]), mapping)
    conv2, conv3 = (estimator.workload.get_node_by_name(n) for n in ("Conv2", "Conv3"))
    assert conv2.tensors[0].shape == conv3.tensors[0].shape and conv2.outputs[0].shape == conv3.outputs[0].shape
    assert not conv2.has_same_performance(conv3)


@pytest.mark.slow
def test_zigzag_averages_the_tiles_it_costs_apart(tiled):
    """Conv1 of the unpadded chain costs the mean of its 7- and 9-column tiles, each ZigZag's evaluation of it."""
    _, estimator = tiled(_valid(), _mapping("{dim: Conv2.D1, tile: 7}"))
    node = estimator.workload.get_node_by_name("Conv1")
    core = estimator.mapping.get(node).resource_allocation[0][0]
    apart = [
        (estimator.run_zigzag(estimator.get_layer_node(node, core, sizes), core), share)
        for sizes, share in estimator.blocks(node, core)
    ]
    entry = estimator.estimate(node, core)
    assert entry.latency_total == pytest.approx(sum(estimator.steady_latency(c) * w for c, w in apart))
    assert entry.energy_total == pytest.approx(sum(c.energy_total * w for c, w in apart))


def test_cutting_a_reader_along_an_axis_it_leaves_its_producer_names_both_and_the_range(tiled):
    """Conv2 (1x1, pad 1) makes 14 rows from Conv1's 12, fusing only for 14: tiling its rows into 2 blocks in a group
    holding both cannot map a tile of Conv2 onto one of Conv1."""
    with pytest.raises(
        ValueError,
        match="reads axis 2 of Conv1's output, of extent 12, along it, which fuses the two "
        "only for an extent from 14 to 14",
    ):
        tiled(_refused(), _mapping("{dim: Conv2.D2, tile: 7}"))


def test_a_group_keeping_the_readers_axis_whole_only_warns(tiled, caplog):
    """Tiling Conv2's 8 output channels into 2 blocks cuts neither axis Conv1 keeps: the group tiles, with the
    warning naming both nodes."""
    with caplog.at_level(logging.WARNING, logger="stream.workload.workload"):
        stage, _ = tiled(_refused(), _mapping("{dim: Conv2.D6, tile: 4}"))
    assert list(stage.fusion_splits.values()) == [2]
    assert "Conv2 reads an axis of Conv1 of extent 12 for 14 outputs" in caplog.text
