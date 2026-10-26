from typing import Any

import pytest
from zigzag.utils import open_yaml

from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.ports import OUTPUT, READ, READ_BY_DATAPATH, WRITE, input_role
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.estimation.core_cost_backends import ZIGZAG_BACKEND, port_traffic
from stream.stages.parsing.accelerator_parser import parse_accelerator

FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
AIE2 = "stream/inputs/aie/hardware/whole_array.yaml"
VIEWS = "tests/fixtures/hardware/tpu_v7_views.yaml"
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
    sram = [port for port in registry if port.key.memory == "sram"]
    assert {port.key for port in sram} == {("sram", "r_port_1"), ("sram", "w_port_1")}
    assert all(port.core_ids == (0, 1) for port in sram)
    assert registry.ports_of(fusemax.get_core(0)) == registry.ports_of(fusemax.get_core(1))


def test_each_operand_gets_its_own_dram_port(fusemax: Accelerator):
    """The one link to DRAM passes all three of its ports; which one a transfer takes is the operand's."""
    registry, dram = fusemax.ports, fusemax.offchip_memory
    assert all(port.core_ids == () for port in registry if port.key.memory == "dram")
    links = [d["cl"] for u, v, d in fusemax.network.edges(data=True) if dram in (u, v)]
    assert {link.ports[dram] for link in links} == {("rw_port_1", "rw_port_2", "rw_port_3")}
    roles = (input_role(1), input_role(2), OUTPUT)
    for direction in (READ, WRITE):
        names = {role: registry.memory_port(dram, direction, role, links) for role in roles}
        assert {role: port and port.key.port for role, port in names.items()} == {
            "input1": "rw_port_1",
            "input2": "rw_port_2",
            "output": "rw_port_3",
        }


def test_declared_port_width_becomes_the_port_model(fusemax: Accelerator):
    # fusemax.yaml: DRAM's rw ports of 3200 b/cc
    port = next(p for p in fusemax.ports if p.key == ("dram", "rw_port_1"))
    assert port.bits_per_cycle == 3200
    assert port.bandwidth.efficiency(4, "read") == 1.0


def test_aie2_tiles_expose_their_dma_channels_as_ports():
    accelerator = parse_accelerator(AIE2)
    compute, mem_tile = accelerator.get_core(2), accelerator.get_core(1)
    rates = {(p.key.port, p.bits_per_cycle) for p in accelerator.ports.ports_of(compute)}
    assert rates == {("mm2s", 2 * 64), ("s2mm", 2 * 64)}
    assert accelerator.ports.port_for(mem_tile, WRITE, input_role(1)).bits_per_cycle == 6 * 64
    assert accelerator.ports.port_for(compute, READ_BY_DATAPATH, input_role(1)) is None


def test_a_measured_memory_has_no_ports():
    accelerator = fusemax_accelerator({"dram": MEASURED})
    registry = accelerator.ports
    assert {port.key.memory for port in registry} == {"sram"}
    assert set(accelerator.bandwidth) == {accelerator.memories["dram"]}


def test_a_port_key_replaces_that_ports_declared_width():
    accelerator = fusemax_accelerator({"dram.rw_port_1": MEASURED, "sram.r_port_1": MEASURED})
    ports = {port.key: port for port in accelerator.ports}
    measured = BandwidthModel.from_description(MEASURED)
    assert ports[("dram", "rw_port_1")].bandwidth == measured
    assert ports[("sram", "r_port_1")].bandwidth == measured
    assert ports[("dram", "rw_port_2")].bits_per_cycle == 3200
    assert accelerator.bandwidth == {}


@pytest.mark.parametrize("key", [2, "hbm", "hbm.rw_port_1", "2.rw_port_1", "dram.dram.rw_port_1"])
def test_bandwidth_keys_naming_no_memory_are_rejected(key: Any):
    assert not fusemax_with_bandwidth({key: MEASURED})[0]


@pytest.mark.parametrize("key", ["dram.rw_port_9", "sram.rw_port_1"])
def test_port_keys_naming_no_declared_port_are_rejected_at_parse_time(key: Any):
    with pytest.raises(ValueError, match="no memory declares"):
        fusemax_accelerator({key: MEASURED})


def test_an_aie2_port_key_naming_no_declared_port_is_rejected_at_parse_time():
    data = open_yaml(AIE2)
    data["bandwidth"] = {"dram.port": MEASURED}
    validator = AcceleratorValidator(data, AIE2)
    assert validator.validate()
    with pytest.raises(ValueError, match="no memory declares"):
        AcceleratorFactory(validator.normalized_data).create()


def test_a_port_key_on_a_measured_memory_is_rejected():
    assert not fusemax_with_bandwidth({"dram": MEASURED, "dram.rw_port_1": MEASURED})[0]


class _BackendWithoutPortTraffic:
    name = "bare"
    priority = 0


def test_a_cost_backend_without_port_traffic_reports_none():
    entry = CoreCostEntry(energy_total=0.0, latency_total=0.0, ideal_cycle=0.0, ideal_temporal_cycle=0.0)
    assert port_traffic(_BackendWithoutPortTraffic(), entry) == ()
    assert port_traffic(ZIGZAG_BACKEND, entry) == ()


def test_views_of_one_memory_share_its_ports():
    # tpu_v7_views.yaml: `8.vmem`, `0.operand_buffer` and `1.operand_buffer` are one scratchpad.
    accelerator = parse_accelerator(VIEWS)
    vmem = [port for port in accelerator.ports if 8 in port.core_ids]
    assert {port.key for port in vmem} == {("vmem_0", "r_port_1"), ("vmem_0", "w_port_1")}
    assert all(port.core_ids == (0, 1, 8) for port in vmem)
    assert all(port.bits_per_cycle == 262144 for port in vmem)
    assert accelerator.ports.ports_of(accelerator.get_core(0)) == tuple(vmem)


@pytest.mark.parametrize(
    "malformed",
    [
        lambda data: data["memories"]["dram"]["ports"]["rw_port_1"].pop("serves"),
        lambda data: data["memories"]["sram"]["ports"]["r_port_1"].pop("cores"),
        lambda data: data["memories"]["sram"]["ports"]["r_port_1"].update(bandwidth=3),
        lambda data: data["memories"]["dram"]["ports"]["rw_port_1"].update(serves={"weights": ["read"]}),
        lambda data: data["memories"]["dram"]["ports"]["rw_port_1"].update(serves={"any": ["fetch"]}),
        lambda data: data["memories"]["sram"].pop("offchip"),
        lambda data: data["memories"]["sram"].pop("ports"),
        lambda data: data["memories"].update(sram=["0.sram", "1.sram"]),
        lambda data: data.update(offchip_core_id=0),
    ],
    ids=[
        "own port without serves",
        "view port without cores",
        "view and own port mixed",
        "unknown operand role",
        "unknown direction",
        "no offchip flag",
        "no ports",
        "views alone",
        "offchip core",
    ],
)
def test_a_memory_written_out_in_part_is_rejected(malformed: Any):
    data = open_yaml(FUSEMAX)
    malformed(data)
    assert not AcceleratorValidator(data, FUSEMAX).validate()
