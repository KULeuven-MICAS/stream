from typing import Any

import pytest
from zigzag.utils import open_yaml

from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.ports import OUTPUT, READ_BY_DATAPATH, WRITE, WRITE_BY_DATAPATH, input_role
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.estimation.core_cost_backends import ZIGZAG_BACKEND, port_traffic
from stream.stages.parsing.accelerator_parser import parse_accelerator

FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
AIE2 = "stream/inputs/aie/hardware/whole_array.yaml"
ALIASED = "tests/fixtures/hardware/tpu_v7_aliased.yaml"
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
    registry = fusemax.ports
    sram = [port for port in registry if port.memory == "sram"]
    assert {port.key for port in sram} == {(0, "sram", "r_port_1"), (0, "sram", "w_port_1")}
    assert all(port.core_ids == (0, 1) for port in sram)
    assert registry.ports_of(fusemax.get_core(0)) == registry.ports_of(fusemax.get_core(1))


def test_each_operand_gets_its_own_dram_port(fusemax: Accelerator):
    registry = fusemax.ports
    dram = fusemax.get_core(2)
    assert len(registry.ports_of(dram)) == 3
    roles = (input_role(1), input_role(2), OUTPUT)
    names = {role: registry.port_for(dram, READ_BY_DATAPATH, role) for role in roles}
    assert {role: port and port.name for role, port in names.items()} == {
        "input1": "rw_port_1",
        "input2": "rw_port_2",
        "output": "rw_port_3",
    }


def test_a_direction_the_operand_does_not_use_falls_back_to_a_port_serving_it(fusemax: Accelerator):
    registry = fusemax.ports
    port = registry.port_for(fusemax.get_core(2), WRITE_BY_DATAPATH, input_role(1))
    assert port is not None and port.name == "rw_port_3"


def test_declared_port_width_and_access_energy_become_the_port_model(fusemax: Accelerator):
    # fusemax_dram.yaml: rw ports of 3200 b/cc, r_cost = w_cost = 32000 pJ per access
    port = next(p for p in fusemax.ports if p.key == (2, "dram", "rw_port_1"))
    assert port.bits_per_cycle == 3200
    assert port.bandwidth.efficiency(4, "read") == 1.0
    assert port.read_energy_per_bit == port.write_energy_per_bit == 10.0


def test_aie2_tiles_expose_their_dma_channels_as_ports():
    accelerator = parse_accelerator(AIE2)
    compute, mem_tile = accelerator.get_core(2), accelerator.get_core(1)
    rates = {(p.name, p.bits_per_cycle) for p in accelerator.ports.ports_of(compute)}
    assert rates == {("mm2s", 2 * 64), ("s2mm", 2 * 64)}
    assert accelerator.ports.port_for(mem_tile, WRITE, input_role(1)).bits_per_cycle == 6 * 64
    assert accelerator.ports.port_for(compute, READ_BY_DATAPATH, input_role(1)) is None


def test_a_core_with_measured_bandwidth_has_no_ports():
    accelerator = fusemax_accelerator({2: MEASURED})
    registry = accelerator.ports
    assert registry.ports_of(accelerator.get_core(2)) == ()
    assert {port.memory for port in registry} == {"sram"}
    assert set(accelerator.bandwidth) == {2}


def test_a_port_key_replaces_that_ports_declared_width():
    accelerator = fusemax_accelerator({"2.dram.rw_port_1": MEASURED, "1.sram.r_port_1": MEASURED})
    ports = {port.key: port for port in accelerator.ports}
    measured = BandwidthModel.from_description(MEASURED)
    assert ports[(2, "dram", "rw_port_1")].bandwidth == measured
    assert ports[(0, "sram", "r_port_1")].bandwidth == measured
    assert ports[(2, "dram", "rw_port_2")].bits_per_cycle == 3200
    assert accelerator.bandwidth == {}


@pytest.mark.parametrize("key", [9, "9.dram.rw_port_1", "2.dram"])
def test_bandwidth_keys_naming_no_core_are_rejected(key: Any):
    assert not fusemax_with_bandwidth({key: MEASURED})[0]


@pytest.mark.parametrize("key", ["2.sram.rw_port_1", "2.dram.rw_port_9", "0.rf_I.r_port_1"])
def test_port_keys_naming_no_top_level_port_are_rejected_at_parse_time(key: Any):
    with pytest.raises(ValueError, match="declares no port"):
        fusemax_accelerator({key: MEASURED})


def test_aie2_port_keys_are_rejected_at_parse_time():
    data = open_yaml(AIE2)
    data["bandwidth"] = {"0.mem.port": MEASURED}
    validator = AcceleratorValidator(data, AIE2)
    assert validator.validate()
    with pytest.raises(ValueError, match="models no memory ports"):
        AcceleratorFactory(validator.normalized_data).create()


def test_a_port_key_on_a_measured_core_is_rejected():
    assert not fusemax_with_bandwidth({2: MEASURED, "2.dram.rw_port_1": MEASURED})[0]


class _BackendWithoutPortTraffic:
    name = "bare"
    priority = 0


def test_a_cost_backend_without_port_traffic_reports_none():
    entry = CoreCostEntry(energy_total=0.0, latency_total=0.0, ideal_cycle=0.0, ideal_temporal_cycle=0.0)
    assert port_traffic(_BackendWithoutPortTraffic(), entry) == ()
    assert port_traffic(ZIGZAG_BACKEND, entry) == ()


def test_aliased_memories_share_one_physical_port():
    # tpu_v7_aliased.yaml: `8.vmem`, `0.operand_buffer` and `1.operand_buffer` are one scratchpad.
    accelerator = parse_accelerator(ALIASED)
    vmem = [port for port in accelerator.ports if 8 in port.core_ids]
    assert {port.key for port in vmem} == {(8, "vmem", "r_port_1"), (8, "vmem", "w_port_1")}
    assert all(port.core_ids == (0, 1, 8) for port in vmem)
    assert all(port.bits_per_cycle == 262144 for port in vmem)
    assert accelerator.ports.ports_of(accelerator.get_core(0)) == tuple(vmem)
