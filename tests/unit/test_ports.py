from typing import Any

import pytest
from zigzag.hardware.architecture.memory_port import DataDirection
from zigzag.utils import open_yaml

from stream.cost_model.bandwidth import BandwidthModel
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.ports import PortRegistry
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.parsing.accelerator_parser import parse_accelerator

FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
MEASURED = {
    "ceiling": 296.6,
    "contiguous": 148.3,
    "strided": {"read": {32: 19.0, 512: 93.5}, "write": {32: 17.1, 512: 93.5}},
}


def fusemax_with_bandwidth(bandwidth: dict[Any, Any]) -> tuple[bool, dict[str, Any]]:
    """Validate fusemax with a ``bandwidth:`` section; returns (valid, normalized data)."""
    data = open_yaml(FUSEMAX)
    data["bandwidth"] = bandwidth
    validator = AcceleratorValidator(data, FUSEMAX)
    return validator.validate(), validator.normalized_data


def fusemax_accelerator(bandwidth: dict[Any, Any]) -> Accelerator:
    valid, data = fusemax_with_bandwidth(bandwidth)
    assert valid
    return AcceleratorFactory(data).create()


@pytest.fixture(scope="module")
def fusemax() -> Accelerator:
    return parse_accelerator(FUSEMAX)


def test_cores_sharing_a_memory_share_its_ports(fusemax: Accelerator):
    registry = PortRegistry.from_accelerator(fusemax)
    sram = [port for port in registry if port.memory == "sram"]
    assert {port.key for port in sram} == {(0, "sram", "r_port_1"), (0, "sram", "w_port_1")}
    assert all(port.core_ids == (0, 1) for port in sram)
    assert registry.ports_of(fusemax.get_core(0)) == registry.ports_of(fusemax.get_core(1))


def test_each_operand_gets_its_own_dram_port(fusemax: Accelerator):
    registry = PortRegistry.from_accelerator(fusemax)
    dram = fusemax.get_core(2)
    assert len(registry.ports_of(dram)) == 3
    names = {op: registry.port_for(dram, DataDirection.RD_OUT_TO_LOW, op) for op in ("I1", "I2", "O")}
    assert {op: port and port.name for op, port in names.items()} == {
        "I1": "rw_port_1",
        "I2": "rw_port_2",
        "O": "rw_port_3",
    }


def test_a_direction_the_operand_does_not_use_falls_back_to_a_port_serving_it(fusemax: Accelerator):
    registry = PortRegistry.from_accelerator(fusemax)
    port = registry.port_for(fusemax.get_core(2), DataDirection.WR_IN_BY_LOW, "I1")
    assert port is not None and port.name == "rw_port_3"


def test_declared_port_width_and_access_energy_become_the_port_model(fusemax: Accelerator):
    # fusemax_dram.yaml: rw ports of 3200 b/cc, r_cost = w_cost = 32000 pJ per access
    port = next(p for p in PortRegistry.from_accelerator(fusemax) if p.key == (2, "dram", "rw_port_1"))
    assert port.bits_per_cycle == 3200
    assert port.bandwidth.efficiency(4, "read") == 1.0
    assert port.read_energy_per_bit == port.write_energy_per_bit == 10.0


def test_aie2_cores_have_no_ports():
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    registry = PortRegistry.from_accelerator(accelerator)
    assert len(registry) == 0
    core = next(iter(accelerator.cores.node_list))
    assert registry.port_for(core, DataDirection.RD_OUT_TO_LOW, "I1") is None


def test_a_core_with_measured_bandwidth_has_no_ports():
    accelerator = fusemax_accelerator({2: MEASURED})
    registry = PortRegistry.from_accelerator(accelerator)
    assert registry.ports_of(accelerator.get_core(2)) == ()
    assert {port.memory for port in registry} == {"sram"}
    assert set(accelerator.bandwidth) == {2}


def test_a_port_key_replaces_that_ports_declared_width():
    accelerator = fusemax_accelerator({"2.dram.rw_port_1": MEASURED, "1.sram.r_port_1": MEASURED})
    ports = {port.key: port for port in PortRegistry.from_accelerator(accelerator)}
    measured = BandwidthModel.from_description(MEASURED)
    assert ports[(2, "dram", "rw_port_1")].bandwidth == measured
    assert ports[(0, "sram", "r_port_1")].bandwidth == measured
    assert ports[(2, "dram", "rw_port_2")].bits_per_cycle == 3200
    assert accelerator.bandwidth == {}


@pytest.mark.parametrize(
    "key",
    [
        9,
        "9.dram.rw_port_1",
        "2.sram.rw_port_1",
        "2.dram.rw_port_9",
        "0.rf_I.r_port_1",
        "2.dram",
    ],
)
def test_bandwidth_keys_naming_no_core_or_top_level_port_are_rejected(key: Any):
    assert not fusemax_with_bandwidth({key: MEASURED})[0]


def test_a_port_key_on_a_measured_core_is_rejected():
    assert not fusemax_with_bandwidth({2: MEASURED, "2.dram.rw_port_1": MEASURED})[0]
