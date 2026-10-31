"""The per-core detail of the steady-state IR on the windowed-convolution scenarios: two fused 3x3 convolutions on
input (1, 8, 32, 32), bf16, under each scenario's fixed mapping."""

from math import prod
from pathlib import Path
from typing import Any

import pytest
from conv_windows import HARDWARE, MAPPINGS, conv, conv_chain, onnx_graph
from onnx import helper

from stream.allocation.allocation import Allocation
from stream.api import SolveOptions, evaluate_mapping
from stream.ir import AllocationIR
from stream.opt.allocation.constraint_optimization.space import DecisionSpace

INTO_CONV2 = "Transfer(conv1_out)"
# Per Conv2 core, the elements each Conv1 core hands it per firing; Conv2's window and what one firing moves.
EXPECTED = {
    "s3": {
        "iterations": 1,
        "cores": ((0, 1, 2, 3), (0, 1, 2, 3)),
        "conv1_tile": [[1, 16, 32, 8]] * 4,
        "out_tile": [[1, 32, 32, 8]] * 4,
        "window": [[1, 16, 32, 9], [1, 16, 32, 10], [1, 16, 32, 10], [1, 16, 32, 9]],
        "moved": [[1, 16, 32, 9], [1, 16, 32, 10], [1, 16, 32, 10], [1, 16, 32, 9]],
        "sources": {
            0: {0: 4096, 1: 512},
            1: {0: 512, 1: 4096, 2: 512},
            2: {1: 512, 2: 4096, 3: 512},
            3: {2: 512, 3: 4096},
        },
    },
    "s4": {
        "iterations": 4,
        "cores": ((0, 1), (2, 3)),
        "conv1_tile": [[1, 16, 8, 16]] * 2,
        "out_tile": [[1, 32, 8, 16]] * 2,
        "window": [[1, 16, 10, 17]] * 2,
        "moved": [[1, 16, 8, 17]] * 2,
        "sources": {2: {0: 2048, 1: 128}, 3: {0: 128, 1: 2048}},
        "conv1_rows": [9, 8, 8, 7],
    },
}


def _allocation(tmp_path: Path, hardware: str, workload: str, mapping: str) -> Allocation:
    options = SolveOptions(artifacts=False)
    return evaluate_mapping(HARDWARE.format(hardware), workload, str(tmp_path), mapping, options).context.get(
        "allocation"
    )


def _steady_state(allocation: Allocation) -> dict[str, Any]:
    return AllocationIR.from_internal(allocation).model_dump()["steady_state"]


def _view(tmp_path: Path, hardware: str, workload: str, mapping: str) -> dict[str, Any]:
    return _steady_state(_allocation(tmp_path, hardware, workload, mapping))


def _core(memory: str) -> int:
    """The eyeriss_like core whose memory, ``eyeriss_like_<id>``, this is."""
    return int(memory.removeprefix("eyeriss_like_"))


def _nodes(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["name"]: node for node in view["tiled_graph"]["nodes"]}


@pytest.mark.parametrize("scenario", ["s3", "s4"])
def test_the_detail_reports_the_tiles_windows_halos_and_sources(tmp_path: Path, scenario: str):
    view = _view(tmp_path, "eyeriss_like_quad_core", conv_chain(), str(MAPPINGS / f"{scenario}.yaml"))
    nodes, expected = _nodes(view), EXPECTED[scenario]
    into2 = nodes[INTO_CONV2]
    handed: dict[int, dict[int, int]] = {}
    for hand in into2["handed"]:
        handed.setdefault(_core(hand["target"]), {})[_core(hand["source"])] = hand["elements"]
    assert view["iterations"] == expected["iterations"]
    assert (tuple(nodes["Conv1"]["cores"]), tuple(nodes["Conv2"]["cores"])) == expected["cores"]
    assert (nodes["Conv1"]["tiles"], nodes["Conv2"]["tiles"]) == (expected["conv1_tile"], expected["out_tile"])
    assert (into2["windows"], into2["moved"]) == (expected["window"], expected["moved"])
    assert handed == expected["sources"]
    assert {(h["type"], h["halo"]) for h in into2["halos"]} >= {("spatial", 2)}
    assert into2["links"] and into2["route_cycles"] > 0
    moved = [prod(tile) for tile in into2["moved"]]
    assert moved == [sum(expected["sources"][_core(memory)].values()) for memory in into2["targets"]]
    assert {axis["dim"] for axis in view["axes"] if axis["sliding"]} >= {h["dim"] for h in into2["halos"]}
    assert nodes["Conv1"]["waited_on"]
    if expected["iterations"] > 1:
        assert next(iter(nodes["Conv1"]["work"].values())) == expected["conv1_rows"]
        assert nodes["Conv1"]["warmup"] > 0
    else:
        assert not any(node["work"] for node in nodes.values())


def test_every_tensor_reports_its_placement_and_buffering(tmp_path: Path):
    view = _view(tmp_path, "eyeriss_like_quad_core", conv_chain(), str(MAPPINGS / "s4.yaml"))
    assert view["tensors"]
    for tensor in view["tensors"]:
        assert tensor["placement"] and tensor["bits"] == 16 * prod(tensor["shape"])
        assert tensor["depth"] is not None and tensor["reuse_level"] is not None


@pytest.mark.parametrize(
    ("scenario", "hardware", "pool"), [("s2", "eyeriss_like_quad_core", False), ("pool_shared", "fusemax", True)]
)
def test_a_copy_within_one_memory_moves_nothing_over_links(tmp_path: Path, scenario: str, hardware: str, pool: bool):
    view = _view(tmp_path, hardware, conv_chain(pool=pool), str(MAPPINGS / f"{scenario}.yaml"))
    into2 = _nodes(view)[INTO_CONV2]
    assert into2["route_cycles"] == 0 and into2["moved_bits"] == 0
    assert into2["moved"] == [[1, 16, 8, 32]]


def test_a_multicast_to_two_readers_reports_each_targets_window(tmp_path: Path):
    """Two 1x1 convolutions, each split over two cores, read one input: its transfer reaches four cores, each
    holding and receiving its own half of the columns."""
    (tmp_path / "mapping.yaml").write_text(
        """
layers:
- {name: ConvA, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D1, split: 2}]]}
- {name: ConvB, core_allocation: [[2, 3]], inter_core_tiling: [[{dim: D1, split: 2}]]}
- {name: Add, core_allocation: [[5]]}
fused_groups:
- {name: g, layers: [ConvA, ConvB, Add], intra_core_tiling: [{dim: ConvA.D2, tile: 8}]}
"""
    )
    workload = onnx_graph(
        [
            conv("ConvA", "input", "wa", "a", k=1, pads=0),
            conv("ConvB", "input", "wb", "b", k=1, pads=0),
            helper.make_node("Add", ["a", "b"], ["o"], name="Add"),
        ],
        {"input": [1, 8, 32, 32], "wa": [16, 8, 1, 1], "wb": [16, 8, 1, 1]},
        ["o"],
    )
    load = _nodes(_view(tmp_path, "eyeriss_like_quad_core", workload, str(tmp_path / "mapping.yaml")))[
        "Transfer(input)"
    ]
    assert [_core(memory) for memory in load["targets"]] == [0, 1, 2, 3]
    assert load["windows"] == load["moved"] == [[1, 8, 8, 16]] * 4


def test_readers_with_other_windows_on_the_same_cores_each_report_their_copy(tmp_path: Path):
    """A 1x1 and a 3x3 convolution split over the same two cores read one input: each core holds both copies, the 3x3
    one with its halo."""
    (tmp_path / "mapping.yaml").write_text(
        """
layers:
- {name: ConvA, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D1, split: 2}]]}
- {name: ConvB, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D1, split: 2}]]}
- {name: Add, core_allocation: [[5]]}
fused_groups:
- {name: g, layers: [ConvA, ConvB, Add], intra_core_tiling: [{dim: ConvA.D2, tile: 8}]}
"""
    )
    workload = onnx_graph(
        [
            conv("ConvA", "input", "wa", "a", k=1, pads=0),
            conv("ConvB", "input", "wb", "b", k=3, pads=1),
            helper.make_node("Add", ["a", "b"], ["o"], name="Add"),
        ],
        {"input": [1, 8, 32, 32], "wa": [16, 8, 1, 1], "wb": [16, 8, 3, 3]},
        ["o"],
    )
    load = _nodes(_view(tmp_path, "eyeriss_like_quad_core", workload, str(tmp_path / "mapping.yaml")))[
        "Transfer(input)"
    ]
    assert load["copies"] == 2 and [_core(memory) for memory in load["targets"]] == [0, 1, 0, 1]
    assert load["windows"] == [[1, 8, 8, 16]] * 2 + [[1, 8, 10, 17]] * 2


def test_a_node_whose_detail_fails_keeps_its_summary_and_the_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    allocation = _allocation(tmp_path, "eyeriss_like_quad_core", conv_chain(), str(MAPPINGS / "s2.yaml"))
    overlaps = DecisionSpace.overlaps

    def failing(space: DecisionSpace, tr: Any) -> Any:
        if tr.name == "Transfer(w2)":
            raise RuntimeError("unreadable transfer")
        return overlaps(space, tr)

    monkeypatch.setattr(DecisionSpace, "overlaps", failing)
    nodes = _nodes(_steady_state(allocation))
    assert nodes["Transfer(w2)"]["kind"] == "transfer" and "handed" not in nodes["Transfer(w2)"]
    assert "handed" in nodes[INTO_CONV2] and nodes["Conv1"]["cores"]
