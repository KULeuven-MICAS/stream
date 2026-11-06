"""ZigZag compute cores as AccelForge arches: structure, energies and bandwidths, on the arch and on mappings."""

from __future__ import annotations

import math

import pytest
import yaml
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.hardware.architecture.core import Core
from stream.stages.estimation.accelforge.tree import PJ
from stream.stages.estimation.accelforge.workload import EinsumDescription, describe
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

from .cores import COMPUTE_CORES, CORES, example_core, load_core, matmul

arch = pytest.importorskip("stream.stages.estimation.accelforge.arch", reason="accelforge is not installed")
mapper = pytest.importorskip("stream.stages.estimation.accelforge.mapper", reason="accelforge is not installed")
accelforge = pytest.importorskip("accelforge")

MATMUL = describe(matmul(), (64, 64, 64))


def _yaml(name: str) -> dict:
    return yaml.safe_load((CORES / f"{name}.yaml").read_text())


def _nodes(core: Core) -> dict:
    return {str(node.name): node for node in arch.core_to_arch(core).nodes}


def _energy_optimal(core: Core, einsum: EinsumDescription, free_memory_order: bool = False):
    spec = arch.explicit_spec(arch.core_to_arch(core, free_memory_order=free_memory_order), mapper.to_workload(einsum))
    spec.mapper.metrics = accelforge.Metrics.ENERGY
    return spec.map_workload_to_arch(print_progress=False)


def _reads(core: Core, einsum: EinsumDescription) -> dict:
    """Per memory, its read action evaluated for ``einsum``."""
    spec = arch.explicit_spec(arch.core_to_arch(core), mapper.to_workload(einsum))
    (path,) = spec._spec_eval_expressions(einsum_name=einsum.name)._get_flattened_architecture()
    return {str(node.name): node.actions["read"] for node in path if hasattr(node, "tensors")}


def _values_per_read(core: Core) -> dict[str, dict[str, float]]:
    reads = _reads(core, MATMUL)
    return {name: {str(k): float(v) for k, v in read.values_per_action.items()} for name, read in reads.items()}


def test_tpu_like_becomes_a_tree_with_streams_energies_and_bandwidths():
    nodes = _nodes(example_core("tpu_like"))
    assert list(nodes) == ["sram_2MB", "array_D1", "rf_4B", "array_D2", "rf_128B", "MAC"]
    assert [(s.name, s.fanout) for n in nodes.values() for s in getattr(n, "spatial", ())] == [("D1", 32), ("D2", 32)]
    sram, accumulator, register = nodes["sram_2MB"], nodes["rf_4B"], nodes["rf_128B"]
    assert math.isinf(sram.size)  # the operands' top memory, unbounded as ZigZag treats it
    assert (accumulator.size, register.size) == (32, 1024)
    read, write = sram.actions["read"], sram.actions["write"]
    assert (read.bits_per_action, read.throughput) == (64, 2048 / 64)  # 64..2048-bit port, one per cycle
    assert read.energy == pytest.approx(416.16 * PJ * 64 / 2048, rel=1e-12)
    assert write.energy == pytest.approx(378.4 * PJ * 64 / 2048, rel=1e-12)
    assert accumulator.actions["read"].throughput == 2  # two 32-bit read ports summed
    assert dict(accumulator.bits_per_value) == {"partial_sums": 32}  # int32 accumulator below the output's top
    assert nodes["MAC"].actions["compute"].energy == pytest.approx(0.04 * PJ)
    assert str(sram.tensors.keep) == "input | weight | output | ~(input | weight | output)"
    assert str(register.tensors.keep) == "weight"


@pytest.mark.parametrize("name", COMPUTE_CORES)
def test_every_example_compute_core_maps_a_matmul_at_its_mac_energy(name):
    core = example_core(name)
    result = _energy_optimal(core, MATMUL)
    assert result.actions(per_component=True)[("MAC", "compute")] == 64**3
    mac_energy = result.energy(per_component=True)["MAC"]
    assert mac_energy == pytest.approx(64**3 * core.operational_array.unit.energy_cost * PJ)


def test_free_memory_order_reaches_a_cheaper_mapping_as_zigzag_does():
    core = example_core("fusemax_array")
    forced, free = (float(_energy_optimal(core, MATMUL, free).energy()) for free in (False, True))
    assert free < 0.95 * forced  # 833126 against 910950 pJ


def test_memories_whose_served_dimensions_do_not_nest_are_copied():
    # wreg serves D3 and result_buffer serves D2: D3 above D2 copies wreg over D3 (2) instead of result_buffer (256).
    core = example_core("tpu_v7_mxu")
    nodes = _nodes(core)
    assert list(nodes) == ["vmem", "array_D1", "array_D3", "result_buffer", "array_D2", "wreg", "MAC"]
    wreg = next(level for level in core.memory_hierarchy.mem_level_list if level.name == "wreg")
    assert nodes["wreg"].size == wreg.memory_instance.size // 2


def test_one_value_periods_pay_whole_accesses_and_cycles_and_wider_periods_pay_by_the_bit():
    data = _yaml("eyeriss_like")
    for port in data["memories"]["rf_64B_A"]["ports"]:  # a 4-bit port: each 8-bit input takes two accesses
        port["bandwidth_min"] = port["bandwidth_max"] = 4
    for port in data["memories"]["rf_16B"]["ports"]:  # a 16-bit port: each 24-bit partial sum takes two
        port["bandwidth_min"] = port["bandwidth_max"] = 16
    data["operand_precision"]["accumulator"] = "int24"
    values = _values_per_read(load_core(data))
    assert (values["rf_64B_A"], values["rf_16B"]) == ({"A": 0.5}, {"C": 0.5})
    assert values["sram_1M"] == values["sram_8KB"] == {}
    # fusemax rf_I: 8-bit accesses on a 16-bit port; ZigZag gives one 8-bit value a whole cycle.
    narrow = _reads(example_core("fusemax_array"), MATMUL)["rf_I"]
    assert float(narrow.throughput) * float(narrow.throughput_scale) == 1


def test_a_memory_feeding_several_units_costs_what_zigzag_charges():
    port = {"name": "rw_port_1", "type": "read_write", "bandwidth_min": 24, "bandwidth_max": 48}
    port["allocation"] = ["I1, tl", "I2, tl", "O, fl", "O, tl", "O, fh", "O, th"]
    memory = {"size": 10**7, "r_cost": 1.0, "w_cost": 1.0, "area": 0, "latency": 1, "operands": ["I1", "I2", "O"]}
    core = load_core(
        {
            "name": "three_units",
            "type": "zigzag.compute",
            "memories": {"buf": memory | {"ports": [port], "served_dimensions": ["D1"]}},
            "operational_array": {"unit_energy": 0.0, "unit_area": 0.01, "dimensions": ["D1"], "sizes": [3]},
            "operator_types": ["Add"],
        }
    )
    a, b, c = (Tensor.create(name, bf16, (96,)) for name in "ABC")
    maps = tuple(AffineMap.from_callable(lambda d: (d,)) for _ in range(3))
    node = ComputationNode(type="Add", name="add", inputs=(a, b), outputs=(c,), operand_mapping=maps)
    edges = [InEdge(name="a", outputs=(a,)), InEdge(name="b", outputs=(b,)), OutEdge(name="c", inputs=(c,))]
    zigzag = ZigZagCostEstimator(Workload([*edges[:2], node, edges[2]]), None, None).estimate(node, core)
    result = _energy_optimal(core, describe(node, (96,)))
    # Three 16-bit values a period through a 24-bit port: two whole accesses, as many as their bits fill.
    assert float(result.energy()) / PJ == pytest.approx(zigzag.energy_total) == 96.0
    assert result.latency(per_component=True)["buf"] == pytest.approx(96.0)  # 4608 bits at 48 bits per cycle
