"""The roofline of the example accelerators, and where the groups of a run sit on it."""

from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from stream.api import SolveOptions, default_families, evaluate_mapping
from stream.cost_model.roofline import COMPUTE, LEVELS, _diagnosis, hardware_roofline, run_roofline
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.stages.parsing.accelerator_parser import parse_accelerator

HARDWARE = "stream/inputs/examples/hardware/{}.yaml"


def test_fusemax_has_a_mac_roof_a_vector_roof_and_one_shared_on_chip_memory():
    """The 256x256 array is the MAC roof and the 256x1 unit the vector roof; both read one shared SRAM."""
    roofline = hardware_roofline(parse_accelerator(HARDWARE.format("fusemax")))
    assert [(r["name"], r["ops_per_cycle"], r["cores"]) for r in roofline["roofs"]] == [
        ("MAC", 65536, [0]),
        ("vector@1", 256, [1]),
    ]
    levels = {level["name"]: level for level in roofline["levels"]}
    assert list(levels) == ["off-chip memory", "off-chip links", "on-chip memory"]
    assert {tuple(r["cores"]) for r in levels["on-chip memory"]["resources"]} == {(0, 1)}
    # its one off-chip link, each way
    assert levels["off-chip links"]["bytes_per_cycle"] == 2 * 512 / 8


def test_vector_cores_serving_other_operators_are_separate_roofs():
    """Eyeriss's 72-wide pooling core and 64-wide elementwise core each bound only the operators they serve."""
    roofs = hardware_roofline(parse_accelerator(HARDWARE.format("eyeriss_like_quad_core")))["roofs"]
    assert [(r["name"], r["ops_per_cycle"]) for r in roofs] == [("MAC", 672), ("vector@4", 72), ("vector@5", 64)]


def test_aie_tiles_have_a_mac_and_a_vector_roof_and_keep_an_on_chip_measured_bandwidth_on_chip():
    """Each of the 32 tiles adds its library's 8x8x8 MAC tile and 16-wide vector rate; a measured bandwidth counts as
    off-chip memory only on the off-chip memory."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array_strix.yaml")
    accelerator.bandwidth[accelerator.memories["tile_0_1"]] = accelerator.bandwidth[accelerator.offchip_memory]
    roofline = hardware_roofline(accelerator)
    assert [(r["name"], r["ops_per_cycle"]) for r in roofline["roofs"]] == [("MAC", 32 * 512), ("vector", 32 * 16)]
    levels = {level["name"]: {r["resource"] for r in level["resources"]} for level in roofline["levels"]}
    assert "dram@0,6,12,18,24,30,36,42" in levels["off-chip memory"] and "tile_0_1@1" in levels["on-chip memory"]


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
    assert point["diagnosis"][0]["code"] in ("compute_bound", "memory_bound", "array_underfilled")


def test_bus_traffic_to_off_chip_memory_reaches_the_off_chip_links(roofline: dict[str, Any]):
    """The quad core reaches its off-chip memory over a bus, whose traffic counts on the off-chip links level."""
    (point,) = roofline["points"]
    (links,) = [level for level in point["levels"] if level["name"] == "off-chip links"]
    assert links["bytes"] > 0 and links["busiest"]["resource"].startswith("bus ")


def test_an_aie_gemm_has_a_roofline_point(tmp_path: Path):
    """The shim's measured bandwidth is the off-chip memory, and kernels without a spatial utilization set no
    ceiling."""
    workload = make_gemm_workload(128, 128, 128, "bf16", "bf16")
    mapping = make_gemm_mapping(128, 128, 128, 64, 64, 64, nb_rows_to_use=2, nb_cols_to_use=2)
    options = SolveOptions(artifacts=False, nb_cols_to_use=2)
    estimate = evaluate_mapping(
        "stream/inputs/aie/hardware/whole_array_strix.yaml", workload, str(tmp_path), mapping, options
    )
    point = estimate.context.get("group_allocations")[0]["roofline"]
    assert point is not None and point["bound"] == "off-chip memory" and point["limit"] > 0
    assert {level["name"]: level["bytes"] > 0 for level in point["levels"]} == {
        "off-chip memory": True,
        "off-chip links": True,
        "on-chip memory": True,
        "core-to-core links": True,
    }


def _point(ops_per_cycle: float, utilization: float) -> dict[str, Any]:
    """A point bound by compute at 100 operations per cycle, whose on-chip memory runs at ``utilization``."""
    busiest = {"resource": "sram.rw@0", "utilization": utilization}
    level = {"name": "on-chip memory", "bytes": 1.0, "intensity": 1.0, "attainable": 200.0, "busiest": busiest}
    return {
        "limiter": COMPUTE,
        "limit": 100.0,
        "ops_per_cycle": ops_per_cycle,
        "latency_cycles": 10,
        "fill_cycles": 4,
        "levels": [level],
    }


def test_a_point_far_below_its_ceiling_on_a_saturated_resource_is_diagnosed_both_ways():
    """A saturated level other than the limiter is named, and so is a gap of more than half below the limit."""
    diagnosis = _diagnosis(_point(ops_per_cycle=20.0, utilization=0.97))
    assert [d["code"] for d in diagnosis] == ["compute_bound", "resource_saturated", "below_ceiling"]
    assert diagnosis[-1] == {"code": "below_ceiling", "fraction": 0.2, "fill_share": 0.4}
    assert [d["code"] for d in _diagnosis(_point(ops_per_cycle=80.0, utilization=0.5))] == ["compute_bound"]


def test_the_summary_splits_the_run_latency_by_bound_and_by_limiter():
    """Two equal groups of 30 cycles and an unplotted one of 40 split the run's 100 cycles 60/40."""

    def point(signature: str, latency: int, bound: str | None, limiter: str | None) -> dict[str, Any]:
        return dict(signature=signature, latency_cycles=latency, macs=1, operations=1, bound=bound, limiter=limiter)

    points = {i: point("a", 30, COMPUTE, "array fill") for i in (0, 1)} | {2: point("b", 40, None, None)}
    summary = run_roofline({}, points, 3)["summary"]
    assert summary["bound_share"] == pytest.approx({COMPUTE: 0.6, "unplotted": 0.4})
    assert summary["limiter_share"] == pytest.approx({"array fill": 0.6, "unplotted": 0.4})


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
    assert (point["roof"], point["macs"], point["operations"]) == ("vector@1", 0.0, 16 * 32 * 32 * 9)
    assert point["ops_per_cycle"] <= 256
    assert estimate.context.get("group_allocations")[0]["roofline"]["roof"] == "vector@1"
