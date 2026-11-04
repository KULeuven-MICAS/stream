"""Workload inputs and outputs placed in an on-chip memory, where they live for the whole run: a compute core with its
own memory l1, a memory core's l2 between it and the off-chip dram, and an on-chip sram no core views."""

from pathlib import Path

import pytest
import yaml
from conv_windows import onnx_graph
from onnx import helper
from placements import offchip_reads, solve

from stream.ir.allocation import AllocationIR
from stream.ir.infeasibility import InfeasibleAllocationError

CORES = Path(__file__).parents[1] / "stream" / "inputs" / "testing" / "hardware" / "cores"


def _port(bandwidth: int, energy: float) -> dict:
    return {"bandwidth": bandwidth, "serves": {"any": ["read", "write"]}, "read_energy": energy, "write_energy": energy}


def _l2_core(bits: int) -> dict:
    """A memory core of one ``bits`` memory, l2, read and written 128 bits a cycle."""
    read, write = (["I1, tl", "I2, tl", "O, tl", "O, th"], ["I1, fh", "I2, fh", "O, fh", "O, fl"])
    ports = [
        {"name": f"{kind[0]}_port_1", "type": kind, "bandwidth_min": 8, "bandwidth_max": 128, "allocation": allocation}
        for kind, allocation in (("read", read), ("write", write))
    ]
    memory = {"size": bits, "r_cost": 60, "w_cost": 60, "area": 0, "latency": 1, "operands": ["I1", "I2", "O"]}
    return {
        "name": "l2_core",
        "type": "zigzag.memory",
        "memories": {"l2": memory | {"ports": ports, "served_dimensions": ["D1", "D2"]}},
        "operational_array": {"unit_energy": 0, "unit_area": 0, "dimensions": ["D1", "D2"], "sizes": [0, 0]},
    }


def _viewed(core: int) -> dict:
    return {port: {"from": port, "cores": [core]} for port in ("r_port_1", "w_port_1")}


def _hardware(tmp_path: Path, l2_bits: int = 2**24, sram_bits: int = 2**23, l2b: bool = False) -> str:
    """Core 0 computes out of l1; core 1 is the memory core of l2, which reaches dram at 8 bits a cycle; sram reaches
    l1 directly; with ``l2b``, core 2 is the memory core of a second such memory, l2b, linked to l1 alone."""
    ends = {
        "l1": ["r_port_1", "w_port_1"],
        "l2": ["r_port_1", "w_port_1"],
        "sram": ["rw_port_1"],
        "dram": ["rw_port_1"],
    }
    data = {
        "name": "cluster",
        "cores": {0: yaml.safe_load((CORES / "tpu_like.yaml").read_text()), 1: _l2_core(l2_bits)},
        "core_coordinates": {0: [0, 0], 1: [0, 1]},
        "unit_energy_cost": 0,
        "memories": {
            "l1": {"offchip": False, "views": ["0.sram_2MB"], "ports": _viewed(0)},
            "l2": {"offchip": False, "views": ["1.l2"], "ports": _viewed(1)},
            "sram": {"offchip": False, "views": [], "capacity_bits": sram_bits, "ports": {"rw_port_1": _port(64, 5)}},
            "dram": {"offchip": True, "views": [], "capacity_bits": 2**33, "ports": {"rw_port_1": _port(8, 100)}},
        },
        "connections": [
            {"type": "link", "ends": {name: ends[name] for name in pair}, "bandwidth": bandwidth}
            for pair, bandwidth in ((("l1", "l2"), 64), (("l2", "dram"), 8), (("l1", "sram"), 64))
        ],
    }
    if l2b:
        data["cores"][2] = _l2_core(l2_bits)
        data["core_coordinates"][2] = [0, 2]
        data["memories"]["l2b"] = {"offchip": False, "views": ["2.l2"], "ports": _viewed(2)}
        data["connections"].append({"type": "link", "ends": {"l1": ends["l1"], "l2b": ends["l2"]}, "bandwidth": 64})
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _placed(*memories: str) -> dict:
    return {"placement": {"memories": list(memories)}}


def _gemm(tmp_path: Path, runtime_args: dict, **hardware):
    """A 64 x 256 by 256 x 256 matmul on core 0, the inputs and outputs placed as ``runtime_args`` says."""
    workload = onnx_graph(
        [helper.make_node("MatMul", ["a", "w"], ["out"], name="MatMul")], {"a": [64, 256], "w": [256, 256]}, ["out"]
    )
    layer = {"name": "MatMul", "core_allocation": [[0]]}
    group = {"name": "g", "layers": ["MatMul"], "intra_core_tiling": []}
    mapping = {"layers": [layer], "fused_groups": [group], "runtime_args": runtime_args}
    return solve(_hardware(tmp_path, **hardware), workload, tmp_path, mapping)


def _mlp(tmp_path: Path, runtime_args: dict, **hardware):
    """Two matmuls in two fused groups on core 0, a 256 KiB weight in the first and a 1 MiB one in the second, the
    hidden tensor crossing the cut."""
    nodes = [
        helper.make_node("MatMul", ["a", "w1"], ["hidden"], name="Up"),
        helper.make_node("MatMul", ["hidden", "w2"], ["out"], name="Down"),
    ]
    workload = onnx_graph(nodes, {"a": [1, 256], "w1": [256, 512], "w2": [512, 1024]}, ["out"])
    layers = [{"name": name, "core_allocation": [[0]]} for name in ("Up", "Down")]
    groups = [{"name": name, "layers": [name], "intra_core_tiling": []} for name in ("Up", "Down")]
    mapping = {"layers": layers, "fused_groups": groups, "runtime_args": runtime_args}
    return solve(_hardware(tmp_path, **hardware), workload, tmp_path, mapping)


def _energy(estimate) -> float:
    return sum(
        AllocationIR(**ir).performance.energy.total_pj for ir in estimate.context.get("group_allocations").values()
    )


def _routes(estimate) -> dict[str, tuple[list[str], list[str]]]:
    """The memories each transfer of the last group reads and writes on its chosen route, by transfer name."""
    allocation = estimate.context.get("allocation")
    routes = (
        (tr.name, allocation.mapping.get(tr).resource_allocation[0])
        for tr in allocation.problem.workload.get_transfer_nodes()
    )
    return {name: ([m.name for m in route.sources], [m.name for m in route.targets]) for name, route in routes}


def _offchip_writes(estimate) -> dict[tuple[str, str], int]:
    return {
        (row["memory"], row["tensor"]): row["write"]
        for group in estimate.context.get("group_memory_accesses").values()
        for row in group["offchip"]
        if row["write"]
    }


@pytest.mark.parametrize("memory", ["l2", "sram", "l1"])
def test_a_weight_placed_on_chip_is_read_there_and_never_from_off_chip(tmp_path, memory):
    estimate = _gemm(tmp_path, {"w": _placed(memory)})
    (ir,) = (AllocationIR(**ir) for ir in estimate.context.get("group_allocations").values())
    assert ir.edge_placements["w"].memories == [memory]
    assert ("dram", "w") not in offchip_reads(estimate)
    assert "w" not in ir.offchip_holdings["dram"]
    assert _routes(estimate)["Transfer(w)"] == ([memory], ["l1"])


def test_a_weight_placed_on_chip_is_faster_and_cheaper_than_off_chip(tmp_path):
    off, on = (_gemm(tmp_path / name, args) for name, args in (("off", {}), ("on", {"w": _placed("l2")})))
    assert offchip_reads(off)[("dram", "w")] > 0
    assert on.context.get("total_latency") < off.context.get("total_latency")
    assert _energy(on) < _energy(off)


def _reads_of(estimate, tensor: str) -> dict[int, int]:
    """Each core's reads of ``tensor``, by core id."""
    (group,) = estimate.context.get("group_memory_accesses").values()
    return {
        core["core_id"]: row["read"] for core in group["cores"] for row in core["tensors"] if row["tensor"] == tensor
    }


@pytest.mark.parametrize(("memory", "core"), [("l2", 1), ("l1", 0)])
def test_a_weight_placed_in_a_core_s_memory_is_read_by_that_core(tmp_path, memory, core):
    """In a memory core's memory, the memory core reads it; in a compute core's own, the compute core."""
    reads = _reads_of(_gemm(tmp_path, {"w": _placed(memory)}), "w")
    assert reads[core] > 0
    assert memory != "l2" or set(reads) == {1}


def test_a_weight_split_over_two_on_chip_memories_is_held_and_read_half_in_each(tmp_path):
    whole = _reads_of(_gemm(tmp_path / "whole", {"w": _placed("l2")}, l2b=True), "w")
    split = {"placement": {"memories": ["l2", "l2b"], "split": 0}}
    estimate = _gemm(tmp_path / "split", {"w": split}, l2b=True)
    assert _reads_of(estimate, "w") == {1: whole[1] // 2, 2: whole[1] // 2}
    assert not offchip_reads(estimate).get(("dram", "w"))
    held = estimate.context.get("allocation").problem.held
    assert {memory.name: {t.name: b for t, b in bits.items()} for memory, bits in held.items()} == {
        "l2": {"w": 2**19},
        "l2b": {"w": 2**19},
    }


def test_a_copy_into_a_memory_holding_one_block_of_a_split_weight_takes_room_for_the_rest(tmp_path):
    """Core 0 reads all of the 128 KiB w, half placed in its l1 and half in sram: the copy into l1 holds the sram half
    beside the placed one."""
    estimate = _gemm(tmp_path, {"w": {"placement": {"memories": ["l1", "sram"], "split": 0}}})
    assert _routes(estimate)["Transfer(w)"] == (["l1", "sram"], ["l1"])
    rows = estimate.context.get("allocation").solution.performance["memory_occupancy"]
    (l1,) = (row for row in rows if row["memory"] == "l1")
    assert sum(row["bits"] for row in l1["tensors"] if row["tensor"].startswith("w")) == 2**20


def test_an_output_and_a_fusion_cut_placed_on_chip_land_there_and_the_next_group_reads_the_cut_there(tmp_path):
    estimate = _mlp(tmp_path, {"Up_cut_out": _placed("l2"), "out": _placed("sram")})
    assert _offchip_writes(estimate) == {}
    assert not {tensor for _, tensor in offchip_reads(estimate)} & {"hidden", "out"}
    assert _routes(estimate)["Transfer(hidden)"] == (["l2"], ["l1"])
    first, second = (AllocationIR(**ir) for ir in estimate.context.get("group_allocations").values())
    assert first.edge_placements["Up_cut_out"] == second.edge_placements["Up_cut_in"]


def test_an_output_placed_on_chip_takes_its_room_there_once(tmp_path):
    """l2 at 192 KiB holds what the matmul stages in it, the 128 KiB w, a and out; out placed there instead fits."""
    l2_bits = 192 * 8 * 2**10
    assert _gemm(tmp_path / "off", {}, l2_bits=l2_bits)
    assert _gemm(tmp_path / "on", {"out": _placed("l2")}, l2_bits=l2_bits)
    with pytest.raises(InfeasibleAllocationError):
        _gemm(tmp_path / "tight", {}, l2_bits=l2_bits - 8 * 2**10)


@pytest.mark.parametrize("memory", ["l2", "sram"])
def test_the_occupancy_report_counts_a_tensor_placed_on_chip(tmp_path, memory):
    estimate = _gemm(tmp_path, {"w": _placed(memory)})
    rows = estimate.context.get("allocation").solution.performance["memory_occupancy"]
    (row,) = (row for row in rows if row["memory"] == memory)
    assert {"tensor": "w", "bits": 2**20} in row["tensors"]
    assert row["resident_bits"] >= 2**20


def test_a_tensor_placed_on_chip_takes_its_room_in_a_group_that_does_not_read_it(tmp_path):
    """l2 holds w1 (256 KiB) and fits either group's own tensors, the second staging the 1 MiB w2, but not w1 beside
    them: the second group, which never reads w1, is refused."""
    l2_bits = (1024 + 128) * 8 * 2**10
    assert _mlp(tmp_path / "off", {}, l2_bits=l2_bits)
    with pytest.raises(InfeasibleAllocationError, match="Memory l2: .* beside w1 placed in it"):
        _mlp(tmp_path / "on", {"w1": _placed("l2")}, l2_bits=l2_bits)


def test_tensors_placed_on_chip_over_its_capacity_are_refused_before_solving(tmp_path):
    with pytest.raises(InfeasibleAllocationError, match=r"^sram holds a \+ w = 160 KiB > 128 KiB$") as refused:
        _gemm(tmp_path, {"a": _placed("sram"), "w": _placed("sram")}, sram_bits=128 * 8 * 2**10)
    (resource,) = refused.value.report.resources
    assert (resource.resource.id, resource.unmet.family) == ("sram", "memory_capacity")
    assert {term.label for term in resource.unmet.terms} == {"a", "w"}


def test_a_placement_both_off_chip_and_on_chip_is_refused(tmp_path):
    with pytest.raises(ValueError, match="off chip and on chip"):
        _gemm(tmp_path, {"w": _placed("l2", "dram")})
