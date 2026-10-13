"""The roofline of the example accelerators, and where the groups of a run sit on it."""

from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from stream.api import SolveOptions, default_families, evaluate_mapping
from stream.cost_model.roofline import COMPUTE, LEVELS, hardware_roofline
from stream.stages.parsing.accelerator_parser import parse_accelerator

HARDWARE = "stream/inputs/examples/hardware/{}.yaml"


def test_fusemax_has_a_mac_roof_a_vector_roof_and_one_shared_on_chip_memory():
    """The 256x256 array is the MAC roof and the 256x1 unit the vector roof; both read one shared SRAM."""
    roofline = hardware_roofline(parse_accelerator(HARDWARE.format("fusemax")))
    assert [(r["name"], r["ops_per_cycle"], r["cores"]) for r in roofline["roofs"]] == [
        ("MAC", 65536, [0]),
        ("vector", 256, [1]),
    ]
    levels = {level["name"]: level for level in roofline["levels"]}
    assert list(levels) == ["off-chip memory", "off-chip links", "on-chip memory"]
    assert {tuple(r["cores"]) for r in levels["on-chip memory"]["resources"]} == {(0, 1)}
    assert levels["off-chip links"]["bytes_per_cycle"] == 4 * 512 / 8


def _onnx(path: Path, nodes: list[onnx.NodeProto], inputs: dict[str, list[int]], output: str, shape: list[int]) -> str:
    graph = helper.make_graph(
        nodes,
        path.stem,
        [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, dims) for name, dims in inputs.items()],
        [helper.make_tensor_value_info(output, TensorProto.BFLOAT16, shape)],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return str(path)


def _blocks(path: Path) -> str:
    """Two identical matmul blocks in a row, which ``per-layer`` cuts into two groups of the same shape."""
    nodes = [
        helper.make_node("MatMul", ["a", "b"], ["c"], name="First"),
        helper.make_node("MatMul", ["c", "d"], ["e"], name="Second"),
    ]
    return _onnx(path, nodes, {n: [64, 64] for n in ("a", "b", "d")}, "e", [64, 64])


def _roofline(tmp: Path, enforce_ports: bool) -> dict[str, Any]:
    accelerator = HARDWARE.format("tpu_like_quad_core")
    ports = {"memory_ports": {"interval": enforce_ports, "burst": enforce_ports}}
    options = SolveOptions(
        artifacts=False,
        families=default_families(accelerator, options=ports),
        stage_options={"fusion_cut_points": "per-layer"},
    )
    return evaluate_mapping(accelerator, _blocks(tmp / "blocks.onnx"), str(tmp), options=options).context.get(
        "roofline"
    )


@pytest.fixture(scope="module")
def roofline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _roofline(tmp_path_factory.mktemp("enforced"), enforce_ports=True)


def test_a_repeated_block_is_one_point_counted_twice(roofline: dict[str, Any]):
    """Both groups run the same shapes on the same cores at the same cost, so they are one point of the whole run."""
    (point,) = roofline["points"]
    assert (point["groups"], point["multiplicity"], point["share"]) == ([0, 1], 2, 1.0)
    assert roofline["summary"]["macs"] == 2 * 64**3
    assert (roofline["groups_solved"], roofline["n_groups"]) == (2, 2)


def test_a_point_sits_on_or_below_every_ceiling_it_is_read_against(roofline: dict[str, Any]):
    """What a group achieves never exceeds the compute roof or what any level's bandwidth lets through."""
    (point,) = roofline["points"]
    ceilings = [level["attainable"] for level in point["levels"] if level["attainable"]]
    assert point["ops_per_cycle"] <= min([*ceilings, point["attainable"]]) * (1 + 1e-9)
    assert point["bound"] in (COMPUTE, *LEVELS)
    for level in point["levels"]:
        if level["bytes"]:
            assert level["intensity"] == pytest.approx(point["operations"] / level["bytes"])
    assert point["diagnosis"][0]["code"] in ("compute_bound", "memory_bound", "array_underfilled")


def test_a_point_above_a_ceiling_names_the_resource_the_solve_did_not_hold_to_its_rate(tmp_path: Path):
    """Ports that only report let the run move more over the off-chip port than its rate allows in that time."""
    (point,) = _roofline(tmp_path, enforce_ports=False)["points"]
    (above,) = [d for d in point["diagnosis"] if d["code"] == "exceeds_ceiling"]
    assert above["level"] == "off-chip memory" and above["utilization"] > 1


def test_a_group_without_macs_on_the_vector_unit_sits_under_the_vector_roof(tmp_path: Path):
    """A max pool alone on FuseMax's vector unit is read in elementwise operations against the 256-wide roof."""
    pool = helper.make_node("MaxPool", ["x"], ["y"], name="Pool", kernel_shape=[3, 3], pads=[1] * 4)
    workload = _onnx(tmp_path / "pool.onnx", [pool], {"x": [1, 16, 32, 32]}, "y", [1, 16, 32, 32])
    accelerator = HARDWARE.format("fusemax")
    options = SolveOptions(artifacts=False, families=default_families(accelerator))
    estimate = evaluate_mapping(accelerator, workload, str(tmp_path), options=options)
    (point,) = estimate.context.get("roofline")["points"]
    assert (point["roof"], point["macs"], point["operations"]) == ("vector", 0.0, 16 * 32 * 32 * 9)
    assert point["ops_per_cycle"] <= 256
    assert estimate.context.get("group_allocations")[0]["roofline"]["roof"] == "vector"
