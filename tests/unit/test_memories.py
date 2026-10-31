"""Tests for physical memories and the cores that view them."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.allocation.lowering import reader_positions
from stream.api import evaluate_mapping
from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.communication_manager import DemandItem
from stream.hardware.architecture.accelerator import Accelerator, MemoryGraph
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import OUTPUT, READ, WRITE, Port, PortKey, input_role
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.mapping.mapping import role_of
from stream.opt.allocation.constraint_optimization.families import available_families
from stream.opt.allocation.constraint_optimization.families.memory import (
    OBJECT_FIFO_DEPTH,
    BufferDescriptors,
    ObjectFifoDepth,
)
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.hardware import build_hardware_facts
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import SolverBackend, SolverParams, SolverVarType, create_solver
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.stages.stage import LeafStage
from stream.visualization.steady_state_trace import _resource_sort_key
from stream.workload.blocks import build_block
from stream.workload.node import ComputationNode, TransferNode
from stream.workload.tensor import Tensor

L3 = """
  l3:
    offchip: true
    views: []
    capacity_bits: 10000000000
    ports:
      rw_port_1: {bandwidth: 64, serves: {any: [read, write]}, read_energy: 15.625, write_energy: 15.625}
"""
TWO_L1S = (
    """
name: two-l1s
cores:
  0: shared_testing_core1.yaml
  1: shared_testing_core2.yaml
  2: shared_testing_core1.yaml
  3: shared_testing_core2.yaml
unit_energy_cost: 0
memories:
  l1_a:
    offchip: false
    views: [0.sram_64KB, 1.sram_64KB]
    ports:
      r_port_1: {from: r_port_1, cores: [0, 1]}
      w_port_1: {from: w_port_1, cores: [0, 1]}
  l1_b:
    offchip: false
    views: [2.sram_64KB, 3.sram_64KB]
    ports:
      r_port_1: {from: r_port_1, cores: [2, 3]}
      w_port_1: {from: w_port_1, cores: [2, 3]}"""
    + L3
    + """connections:
  - {type: link, ends: {l1_a: [r_port_1, w_port_1], l3: [rw_port_1]}, bandwidth: 64}
  - {type: link, ends: {l1_b: [r_port_1, w_port_1], l3: [rw_port_1]}, bandwidth: 64}
  - {type: link, ends: {l1_a: [r_port_1, w_port_1], l1_b: [r_port_1, w_port_1]}, bandwidth: 32}
"""
)
"""Two L1s of two cores each, joined to each other and to L3."""


def _l1(*views: str) -> dict:
    """An on-chip memory with ``views`` of shared testing cores, each port reached by all of them."""
    cores = [int(view.split(".")[0]) for view in views]
    ports = {name: {"from": name, "cores": cores} for name in ("r_port_1", "w_port_1")}
    return {"offchip": False, "views": list(views), "ports": ports}


def _memories(**views: list[str]) -> dict:
    """One on-chip memory per entry of ``views``, plus L3."""
    return {name: _l1(*entries) for name, entries in views.items()} | yaml.safe_load(L3)


WHOLE = ((0, 15), (0, 15))
SHARED_CORE1 = Path("stream/inputs/testing/hardware/cores/shared_testing_core1.yaml")
SHARED_CORE2 = Path("stream/inputs/testing/hardware/cores/shared_testing_core2.yaml")


def _two_l1s() -> dict:
    return yaml.safe_load(TWO_L1S)


def _accelerator(tmp_path, data: dict | None = None, **overrides):
    path = tmp_path / "hw.yaml"
    path.write_text(yaml.safe_dump((data or _two_l1s()) | overrides, sort_keys=False))
    return parse_accelerator(path)


def _invalid(data: dict, path: Path = Path("hw.yaml")) -> list[str]:
    """The errors validating ``data`` reports; it must report some."""
    validator = AcceleratorValidator(data, path)
    assert not validator.validate()
    assert validator.errors
    return validator.errors


def _errors(**overrides) -> list[str]:
    return _invalid(_two_l1s() | overrides)


def test_the_cores_of_a_memory_share_it(tmp_path):
    accelerator = _accelerator(tmp_path)
    a = accelerator.memories["l1_a"]
    assert [(core.id, instance) for core, instance in a.views] == [(0, "sram_64KB"), (1, "sram_64KB")]
    for role in (OUTPUT, input_role(1), input_role(2)):
        assert accelerator.memory_of(accelerator.get_core(0), role) is a
        assert accelerator.memory_of(accelerator.get_core(1), role) is a
    assert accelerator.offchip_memory is accelerator.memories["l3"] and accelerator.offchip_memory.offchip
    assert set(accelerator.network.nodes()) == set(accelerator.memories.values())


def test_any_core_of_a_memory_sends_over_its_links(tmp_path):
    """Each core of one L1 reaches each core of the other over the link between the L1s."""
    accelerator = _accelerator(tmp_path)
    for source, target in ((0, 2), (1, 3), (1, 2)):
        plan, _ = accelerator.communication_manager.route(
            (accelerator.memory_of(accelerator.get_core(source), OUTPUT),),
            (accelerator.memory_of(accelerator.get_core(target), input_role(1)),),
            (DemandItem(0, (0,), WHOLE),),
            16,
            lambda one, other: False,
        )
        assert [str(link) for link in plan.links_used] == ["CL(l1_a, l1_b, bw=32)"]


def test_a_bandwidth_model_is_keyed_by_its_memory(tmp_path):
    accelerator = _accelerator(tmp_path, bandwidth={"l1_a": {"ceiling": 64, "contiguous": 64}})
    assert list(accelerator.bandwidth) == [accelerator.memories["l1_a"]]


@pytest.mark.parametrize("key", [0, "core0", "0.rw_port_1", "l1_a.sram_64KB.rw_port_1"])
def test_a_bandwidth_key_naming_no_memory_is_rejected(key):
    assert _errors(bandwidth={key: {"ceiling": 64, "contiguous": 64}})


def test_a_port_key_of_a_measured_memory_is_rejected():
    model = {"ceiling": 64, "contiguous": 64}
    errors = _errors(bandwidth={"l1_a": model, "l1_a.rw_port_1": model})
    assert any("already has a measured bandwidth" in error for error in errors)


def test_an_instance_in_two_memories_is_rejected():
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["1.sram_64KB", "2.sram_64KB", "3.sram_64KB"]}
    errors = _errors(memories=_memories(**views))
    assert any("`1.sram_64KB` is a view of both l1_a and l1_b" in error for error in errors)


def test_a_core_in_no_memory_is_rejected():
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["2.sram_64KB"]}
    errors = _errors(memories=_memories(**views))
    assert any("core 3 is a view of no memory" in error for error in errors)


def test_a_memory_no_connection_reaches_is_rejected():
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["2.sram_64KB"], "spare": ["3.sram_64KB"]}
    errors = _errors(memories=_memories(**views))
    assert any("spare" in error and "no connection" in error for error in errors)


def test_a_connection_through_a_port_its_memory_does_not_declare_is_rejected():
    links = [{"type": "link", "ends": {"l1_a": ["rw_port_9"], "l3": ["rw_port_1"]}, "bandwidth": 64}]
    errors = _errors(connections=links)
    assert any("ends.l1_a` names ports ['rw_port_9'] it does not declare" in error for error in errors)


def test_a_connection_naming_its_memories_without_their_ports_is_rejected():
    links = [{"type": "link", "memories": ["l1_a", "l3"], "bandwidth": 64}]
    assert _errors(connections=links)


def test_a_connection_naming_cores_is_rejected():
    links = [{"type": "link", "cores": [0, 2], "bandwidth": 64}]
    assert _errors(connections=links)


def test_a_view_of_an_instance_its_core_does_not_declare_is_rejected():
    views = {"l1_a": ["0.sram_64KB", "1.l1"], "l1_b": ["2.sram_64KB", "3.sram_64KB"]}
    assert any("core 1 declares no memory `l1`" in error for error in _errors(memories=_memories(**views)))


def test_a_view_of_an_inner_level_is_rejected():
    """A view must name a top-level instance of its core."""
    text = (
        """
name: inner
cores: {0: testing_core1.yaml}
memories:
  l1: {offchip: false, views: [0.sram_64KB], ports: {}}"""
        + L3
        + """connections: [{type: link, ends: {l1: [], l3: [rw_port_1]}, bandwidth: 64}]
"""
    )
    errors = _invalid(yaml.safe_load(text))
    assert any("`sram_64KB` is no top-level memory; a view is of one of its top levels" in e for e in errors)


def test_a_link_joins_exactly_two_memories():
    links = [{"type": "link", "ends": {"l1_a": [], "l1_b": [], "l3": []}, "bandwidth": 64}]
    assert any("joins exactly two memories" in error for error in _errors(connections=links))


def test_the_ir_lists_each_memory_with_its_views(tmp_path):
    memories = _accelerator(tmp_path).get_ir()["memories"]
    assert memories == {
        "l1_a": {"offchip": False, "views": ["0.sram_64KB", "1.sram_64KB"]},
        "l1_b": {"offchip": False, "views": ["2.sram_64KB", "3.sram_64KB"]},
        "l3": {"offchip": True, "views": []},
    }


def test_a_measured_on_chip_memory_leaves_the_off_chip_traffic_charged(tmp_path):
    """Off-chip traffic is charged unless the off-chip memory itself is measured."""
    family = available_families()["offchip_traffic"]()
    model = {"ceiling": 64, "contiguous": 64}
    for measured, charged in (("l1_a", True), ("l3", False)):
        accelerator = _accelerator(tmp_path, bandwidth={measured: model})
        space = SimpleNamespace(
            accelerator=accelerator, shared_bandwidth=accelerator.bandwidth, iterations=4, offchip_bandwidth=lambda: 64
        )
        quantities = QuantityRegistry()
        family.build(SimpleNamespace(space=space, quantities=quantities))
        assert ("offchip_traffic_weight" in quantities) is charged


META_PROTOTYPE = Path("stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml")


def test_a_core_reads_each_operand_in_the_memory_its_top_level_instance_for_it_is_a_view_of():
    """A meta_prototype core holds activations and outputs in one SRAM and weights in another."""
    accelerator = parse_accelerator(META_PROTOTYPE)
    core = accelerator.get_core(0)
    assert accelerator.memory_of(core, OUTPUT).name == accelerator.memory_of(core, input_role(1)).name
    assert accelerator.memory_of(core, input_role(1)).name == "meta_prototype_0"
    assert accelerator.memory_of(core, input_role(2)).name == "meta_prototype_0_weights"
    assert [m.name for m in accelerator.memories_of(core)] == ["meta_prototype_0", "meta_prototype_0_weights"]
    # a further input is held with the activations
    assert accelerator.memory_holding(core, input_role(3)) is accelerator.memory_of(core, input_role(1))


def test_a_top_level_instance_no_memory_binds_is_rejected():
    """Every top-level instance of a core must be a view of some memory."""
    data = yaml.safe_load(META_PROTOTYPE.read_text())
    del data["memories"]["meta_prototype_0_weights"]
    bus = next(c for c in data["connections"] if c["type"] == "bus")
    del bus["ends"]["meta_prototype_0_weights"]
    errors = _invalid(data, META_PROTOTYPE)
    assert any("core 0's top-level memory `sram_1MB_W` is a view of no memory" in e for e in errors)


def test_weights_land_in_the_weights_memory(tmp_path):
    """Weight copies of a convolution are placed only in the weights SRAM."""
    workload = make_2_conv_workload(
        TwoConvWorkloadConfig(
            batch_size=1,
            in_channels=8,
            height=32,
            width=32,
            out_channels_1=16,
            out_channels_2=32,
            kernel_size=3,
            in_dtype="bf16",
            weight_dtype="bf16",
        )
    )
    allocation = evaluate_mapping(str(META_PROTOTYPE), workload, str(tmp_path)).context.get("allocation")
    workload = allocation.problem.workload
    weights = [
        copy
        for node in workload.get_computation_nodes()
        for copy in node.inputs[1:2]
        if any(isinstance(p, TransferNode) and copy in p.outputs for p in workload.predecessors(node))
    ]
    assert weights
    for copy in weights:
        placed = allocation.solution.tensor_placements[copy]
        assert placed and all(memory.name.endswith("_weights") for memory in placed), (copy.name, placed)


def _two_offchip() -> dict:
    """:data:`TWO_L1S` with a second off-chip memory, ``dram``, declared after ``l3`` and linked to ``l1_b``."""
    data = _two_l1s()
    data["memories"]["dram"] = yaml.safe_load(L3)["l3"]
    data["connections"].append(
        {"type": "link", "ends": {"l1_b": ["r_port_1", "w_port_1"], "dram": ["rw_port_1"]}, "bandwidth": 64}
    )
    return data


def test_the_off_chip_memories_are_those_declared_so_in_declared_order(tmp_path):
    accelerator = _accelerator(tmp_path)
    assert accelerator.offchip_memories == (accelerator.memories["l3"],)
    accelerator = _accelerator(tmp_path, _two_offchip())
    assert [memory.name for memory in accelerator.offchip_memories] == ["l3", "dram"]
    with pytest.raises(ValueError, match="several off-chip memories; use offchip_memories"):
        _ = accelerator.offchip_memory


def test_every_off_chip_link_bounds_the_off_chip_bandwidth(tmp_path):
    accelerator = _accelerator(tmp_path, _two_offchip())
    links = {data["cl"] for _, _, data in accelerator.network.edges(data=True)}
    space = SimpleNamespace(accelerator=accelerator, link_set=links)
    assert DecisionSpace.offchip_bandwidth(space) == 3 * 64


def test_a_memory_declares_its_capacity_only_where_no_core_views_it():
    memories = _two_l1s()["memories"]
    memories["l1_a"]["capacity_bits"] = 8
    del memories["l3"]["capacity_bits"]
    errors = _errors(memories=memories)
    assert any("`memories.l1_a`: its views declare its capacity" in error for error in errors)
    assert any("`memories.l3`: no core views it" in error for error in errors)


def test_a_memory_no_core_views_on_chip_is_a_memory_of_kind_memory(tmp_path):
    """An on-chip memory with no views has kind ``memory`` and no cores."""
    memories = _two_l1s()["memories"]
    memories["scratch"] = {
        "offchip": False,
        "views": [],
        "capacity_bits": 8 * 1024 * 1024,
        "ports": {"rw": {"bandwidth": 64, "serves": {"any": ["read", "write"]}, "read_energy": 0, "write_energy": 0}},
    }
    connections = _two_l1s()["connections"]
    connections.append({"type": "link", "ends": {"scratch": ["rw"], "l1_a": ["r_port_1", "w_port_1"]}, "bandwidth": 64})
    accelerator = _accelerator(tmp_path, memories=memories, connections=connections)
    scratch = accelerator.memories["scratch"]
    assert scratch.kind == "memory" and scratch.cores == () and scratch.namespace is None
    assert accelerator.get_ir()["memories"]["scratch"] == {"offchip": False, "views": []}


def test_a_dma_comes_with_the_ports_that_are_its_channels():
    """A memory declares ``dma`` exactly when one of its ports declares ``channels``."""
    memories = _two_l1s()["memories"]
    memories["l3"]["dma"] = {"buffer_descriptors": 16, "iterations": 64}
    assert any("no port declares `channels`" in error for error in _errors(memories=memories))
    del memories["l3"]["dma"]
    memories["l3"]["ports"]["rw_port_1"]["channels"] = 4
    assert "`memories.l3`: ports ['rw_port_1'] declare `channels`, but no `dma` is given." in _errors(memories=memories)


def test_a_zigzag_accelerator_declares_no_dma():
    """No zigzag family bounds DMA channels, so declaring them is rejected rather than left unbounded."""
    memories = _two_l1s()["memories"]
    memories["l3"]["dma"] = {"buffer_descriptors": 16, "iterations": 64}
    memories["l3"]["ports"]["rw_port_1"]["channels"] = 4
    assert any("`memories.l3`: the zigzag namespace bounds no DMA channels" in e for e in _errors(memories=memories))


def test_an_off_chip_memory_has_no_views():
    memories = _two_l1s()["memories"]
    memories["l1_b"]["offchip"] = True
    del memories["l3"]
    connections = [c for c in _two_l1s()["connections"] if "l3" not in c["ends"]]
    errors = _errors(memories=memories, connections=connections)
    assert "`memories.l1_b` is off-chip, where no core computes, so it has no views." in errors


def _read_only_l3(data: dict) -> None:
    data["memories"]["l3"]["ports"]["rw_port_1"]["serves"] = {"any": ["read"]}


def _portless_l3(data: dict) -> None:
    data["memories"]["l3"]["ports"] = {}
    for connection in data["connections"]:
        if "l3" in connection["ends"]:
            connection["ends"]["l3"] = []


def _l1_a_read_port_to_l3(data: dict) -> None:
    next(c for c in data["connections"] if set(c["ends"]) == {"l1_a", "l3"})["ends"]["l1_a"] = ["r_port_1"]


@pytest.mark.parametrize(
    ("malform", "message"),
    [
        (_read_only_l3, "names no port of memory l3 serving write"),
        (_portless_l3, "names no port of memory l3 serving read or write"),
        (_l1_a_read_port_to_l3, r"CL\(l1_a, l3, bw=64\) names no port of memory l1_a serving write"),
    ],
    ids=["read-only DRAM port", "DRAM without ports", "link end with only a read port"],
)
def test_a_connection_passes_a_port_each_way_at_every_memory_it_joins(tmp_path, malform, message):
    """A transfer over a connection is bounded by a port at each end, whichever way it goes."""
    data = _two_l1s()
    malform(data)
    with pytest.raises(ValueError, match=message):
        _accelerator(tmp_path, data)


def test_a_memory_with_no_route_to_the_off_chip_memory_is_rejected(tmp_path):
    data = _two_l1s()
    ports = yaml.safe_load(L3)["l3"]["ports"]
    data["memories"]["island"] = {"offchip": False, "views": [], "capacity_bits": 1024, "ports": ports}
    data["connections"] = [c for c in data["connections"] if "l1_b" not in c["ends"]]
    data["connections"].append(
        {"type": "link", "ends": {"l1_b": ["r_port_1", "w_port_1"], "island": ["rw_port_1"]}, "bandwidth": 64}
    )
    with pytest.raises(ValueError, match=r"memories \['island', 'l1_b'\] have no route to off-chip memory l3"):
        _accelerator(tmp_path, data)


def test_a_memory_with_a_route_from_one_off_chip_memory_is_reached(tmp_path):
    data = _two_offchip()
    data["connections"] = [c for c in data["connections"] if set(c["ends"]) in ({"l1_a", "l3"}, {"l1_b", "dram"})]
    assert _accelerator(tmp_path, data).offchip_memories
    data["memories"]["island"] = yaml.safe_load(L3)["l3"] | {"offchip": False, "capacity_bits": 1024}
    for connection in data["connections"]:
        if "dram" in connection["ends"]:
            connection["ends"] = {"l1_a": ["r_port_1", "w_port_1"], "dram": ["rw_port_1"]}
    data["connections"].append(
        {"type": "link", "ends": {"l1_b": ["r_port_1", "w_port_1"], "island": ["rw_port_1"]}, "bandwidth": 64}
    )
    with pytest.raises(ValueError, match=r"memories \['island', 'l1_b'\] have no route to off-chip memory l3 or dram"):
        _accelerator(tmp_path, data)


def test_aie_code_generation_refuses_several_off_chip_memories():
    aie_code_generation = pytest.importorskip(
        "stream.stages.codegen.aie_code_generation",
        reason="the AIE dialects are a separate install, via stream-setup-aie",
    )
    path = Path("stream/inputs/aie/hardware/single_core.yaml")
    data = yaml.safe_load(path.read_text())
    data["memories"]["dram_1"] = data["memories"]["dram"]
    data["connections"].append(
        {"type": "link", "ends": {"dram_1": ["mm2s", "s2mm"], "tile_0_1": ["mm2s", "s2mm"]}, "bandwidth": 64}
    )
    validator = AcceleratorValidator(data, path)
    assert validator.validate(), validator.errors
    accelerator = AcceleratorFactory(validator.normalized_data).create()
    with pytest.raises(ValueError, match=r"off-chip memories \['dram', 'dram_1'\]; the AIE runtime sequence"):
        aie_code_generation.AIECodeGenerationStage([LeafStage], StageContext.from_kwargs(accelerator=accelerator))


def test_an_aie2_tile_memory_is_a_view_of_exactly_one_tile():
    """An AIE2 tile memory with two views is rejected."""
    path = Path("stream/inputs/aie/hardware/single_core.yaml")
    data = yaml.safe_load(path.read_text())
    data["memories"]["tile_0_2"]["views"] = ["1.memory", "2.memory"]
    del data["memories"]["tile_0_1"]
    data["connections"] = [conn for conn in data["connections"] if "tile_0_1" not in conn["ends"]]
    errors = _invalid(data, path)
    assert any("`memories.tile_0_2` has 2 views; an AIE2 tile's memory is its own" in e for e in errors)


@pytest.mark.parametrize(
    "malform",
    [lambda core: core["memories"].update(sram_64KB=1048576), lambda core: core.pop("memories")],
    ids=["level written as a number", "no levels"],
)
def test_a_malformed_core_is_an_error_not_a_crash(malform):
    core = yaml.safe_load(SHARED_CORE1.read_text())
    malform(core)
    _errors(cores={**_two_l1s()["cores"], 0: core})


def test_parallel_links_between_two_memories_are_distinct_resources_sorted_by_number(tmp_path):
    """Links between the same memories are told apart by their index, and sort by it as a number."""
    connections = _two_l1s()["connections"]
    connections += [dict(connections[0]) for _ in range(10)]
    accelerator = _accelerator(tmp_path, connections=connections)
    a, b = accelerator.memories["l1_a"], accelerator.memories["l3"]
    links = sorted(d["cl"] for u, v, d in accelerator.network.edges(data=True) if (u, v) == (a, b))
    assert [link.parallel for link in links] == list(range(11)) and len(set(links)) == 11
    assert [str(link) for link in links[:2]] == ["CL(l1_a, l3, bw=64)", "CL#1(l1_a, l3, bw=64)"]


def test_a_port_naming_a_core_twice_is_rejected():
    memories = _two_l1s()["memories"]
    memories["l1_a"]["ports"]["r_port_1"]["cores"] = [0, 0, 1]
    assert any("`memories.l1_a.ports.r_port_1` names a core twice: [0, 0, 1]" in e for e in _errors(memories=memories))


def test_a_memory_between_columns_is_rejected():
    """A memory viewed by cores in different columns is rejected."""
    errors = _errors(core_coordinates={0: [0, 0], 1: [1, 0], 2: [0, 1], 3: [0, 2]})
    assert "`memories.l1_a`: its views' cores sit in different columns [0, 1]." in errors


def _whole_array() -> tuple[Path, dict]:
    path = Path("stream/inputs/aie/hardware/whole_array.yaml")
    return path, yaml.safe_load(path.read_text())


def test_every_aie2_dram_port_is_a_shim_dma():
    """AIE2 DRAM moves data only through shim DMA channels, and the code generator needs their DMA's limits."""
    path, data = _whole_array()
    del data["memories"]["dram"]["ports"]["mm2s_c3"]["channels"]
    assert any("`memories.dram.ports.mm2s_c3` declares no `channels`" in e for e in _invalid(data, path))
    path, data = _whole_array()
    del data["memories"]["dram"]["dma"]
    assert any("`memories.dram`: ports ['mm2s_c0'," in e and "no `dma` is given" in e for e in _invalid(data, path))


def test_an_aie2_shim_dma_serves_one_column():
    path, data = _whole_array()
    to_column_1 = next(c for c in data["connections"] if {"dram", "tile_1_1"} == set(c["ends"]))
    to_column_1["ends"]["dram"] = ["mm2s_c0", "s2mm_c0"]
    errors = _invalid(data, path)
    assert "`memories.dram.ports.mm2s_c0` joins columns [0, 1]; a shim DMA serves its own column." in errors


def test_an_aie2_tile_memory_has_no_channels_of_its_own():
    """A tile memory's DMA channels are its core's; channels on its ports would bound nothing."""
    path, data = _whole_array()
    data["memories"]["tile_0_2"]["ports"]["extra"] = {
        "bandwidth": 64,
        "channels": 2,
        "serves": {"any": ["read"]},
        "read_energy": 0,
        "write_energy": 0,
    }
    errors = _invalid(data, path)
    assert "`memories.tile_0_2`: an AIE2 tile's DMA channels are its core's, not its ports'." in errors


def test_a_staging_memory_compute_cores_compute_out_of_is_of_kind_compute():
    """A staging memory that compute cores also view has kind ``compute``."""
    views = parse_accelerator("tests/fixtures/hardware/tpu_v7_views.yaml")
    vmem = views.memories["vmem_0"]
    assert vmem.staging and vmem.kind == "compute" and vmem.col_id == 0
    assert vmem in build_hardware_facts(views, nb_cols_to_use=4).mem_memories


def test_a_link_not_joining_a_memory_passes_none_of_its_ports():
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    dram, tile = accelerator.offchip_memory, accelerator.memories["tile_0_2"]
    elsewhere = [d["cl"] for u, v, d in accelerator.network.edges(data=True) if u is tile]
    assert accelerator.ports.memory_ports(dram, READ, input_role(1), elsewhere) == {}


def test_two_shim_ports_alike_on_one_link_leave_a_transfer_no_dma():
    """Two DMA ports on one link that both serve a read make the port ambiguous."""
    ports = {
        name: Port(PortKey("dram", name), (), BandwidthModel.flat(128), frozenset({("any", READ)}), channels=2)
        for name in ("shim_b", "shim_a")
    }
    memory = Memory("dram", (), dict(ports), offchip=True, capacity_bits=8)
    with pytest.raises(ValueError, match=r"memory dram: DMA ports \['shim_a', 'shim_b'\] all serve"):
        memory.dma_port(["shim_b", "shim_a"], READ, input_role(1))
    assert memory.dma_port(["shim_b"], READ, input_role(1)).key.port == "shim_b"
    assert memory.dma_port(["shim_b"], WRITE, OUTPUT) is None


def test_a_route_over_two_columns_links_passes_one_shim_port_per_link():
    """A DRAM route to two columns passes one shim port per link."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    dram = accelerator.offchip_memory
    links = [data["cl"] for u, v, data in accelerator.network.edges(data=True) if u is dram and v.col_id in (0, 1)]
    by_port = accelerator.ports.memory_ports(dram, READ, input_role(1), links)
    assert {port.key.port: [link.receiver.name for link in port_links] for port, port_links in by_port.items()} == {
        "mm2s_c0": ["tile_0_1"],
        "mm2s_c1": ["tile_1_1"],
    }
    assert [port.key.port for port in accelerator.ports.memory_ports(dram, READ, input_role(1), links[:1])] == [
        "mm2s_c0"
    ]


def _dram_space(accelerator: Accelerator) -> DecisionSpace:
    """A decision space over ``accelerator`` that reads only the side ports of a first-input transfer."""
    space = object.__new__(DecisionSpace)
    space.accelerator, space.shared_bandwidth = accelerator, {}
    space.operand_role = lambda tr, read_side: input_role(1)  # type: ignore[method-assign]
    return space


@pytest.mark.parametrize(
    ("needs", "shares"),
    [
        ({"tile_0_1": (0, 11)}, {"mm2s_c0": 1}),
        ({"tile_0_1": (0, 11), "tile_1_1": (0, 11)}, {"mm2s_c0": 1, "mm2s_c1": 1}),
        ({"tile_0_1": (0, 11), "tile_1_1": (0, 11), "tile_2_1": (0, 11)}, {"mm2s_c0": 1, "mm2s_c1": 1, "mm2s_c2": 1}),
        ({"tile_0_1": (0, 5), "tile_1_1": (6, 11)}, {"mm2s_c0": 1 / 2, "mm2s_c1": 1 / 2}),
        ({"tile_0_1": (0, 8), "tile_1_1": (9, 11)}, {"mm2s_c0": 3 / 4, "mm2s_c1": 1 / 4}),
        (
            {"tile_0_1": (0, 3), "tile_1_1": (4, 7), "tile_2_1": (8, 11)},
            {"mm2s_c0": 1 / 3, "mm2s_c1": 1 / 3, "mm2s_c2": 1 / 3},
        ),
        ({"tile_0_1": (0, 7), "tile_1_1": (4, 11)}, {"mm2s_c0": 2 / 3, "mm2s_c1": 2 / 3}),
        (
            {"tile_0_1": (0, 5), "tile_1_1": (3, 8), "tile_2_1": (6, 11)},
            {"mm2s_c0": 1 / 2, "mm2s_c1": 1 / 2, "mm2s_c2": 1 / 2},
        ),
        ({"tile_0_2": (0, 7), "tile_0_3": (4, 11), "tile_1_1": (0, 2)}, {"mm2s_c0": 1, "mm2s_c1": 1 / 4}),
    ],
    ids=["one", "broadcast2", "broadcast3", "split2", "uneven2", "split3", "overlap2", "overlap3", "behind"],
)
def test_each_shim_port_carries_what_the_tiles_behind_it_need(needs, shares):
    """A DRAM read passes each shim port it leaves through with the union of what the tiles behind that port need:
    all of a broadcast, its part of a split. Ports work in parallel, so the busiest one paces the side."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    dram = accelerator.offchip_memory
    tiles = tuple(accelerator.memories[name] for name in needs)
    demand = tuple(DemandItem(j, (0,), (rows, (0, 15))) for j, rows in enumerate(needs.values()))
    plan, _ = accelerator.communication_manager.route((dram,), tiles, demand, 16, lambda one, other: False)
    space = _dram_space(accelerator)
    side = {port.key.port: share for port, share in space.side_ports(None, plan, dram, READ).items()}
    assert side == pytest.approx(shares)
    assert space.side_bandwidth(None, plan, dram, READ).ceiling == pytest.approx(128 / max(shares.values()))
    assert side[space.side_port(None, plan, dram, READ).key.port] == max(side.values())


@pytest.mark.parametrize(
    ("needs", "written"),
    [
        ({"tile_0_1": (0, 11), "tile_1_1": (0, 11)}, [1, 1]),
        ({"tile_0_1": (0, 8), "tile_1_1": (9, 11)}, [3 / 4, 1 / 4]),
        ({"tile_0_2": (0, 5), "tile_0_3": (6, 11)}, [1 / 2, 1 / 2]),
    ],
    ids=["broadcast", "split", "passed_on"],
)
def test_each_tile_writes_the_part_of_a_dram_read_it_needs(needs, written):
    """Each tile writes what it needs of the read, not what it passes on: all of a broadcast, its part of a split."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    tiles = tuple(accelerator.memories[name] for name in needs)
    demand = tuple(DemandItem(j, (0,), (rows, (0, 15))) for j, rows in enumerate(needs.values()))
    plan, _ = accelerator.communication_manager.route(
        (accelerator.offchip_memory,), tiles, demand, 16, lambda one, other: False
    )
    space = _dram_space(accelerator)
    assert [sum(space.side_ports(None, plan, tile, WRITE).values()) for tile in tiles] == pytest.approx(written)


def test_a_measured_dram_keeps_its_shim_dma_channels():
    """Strix DRAM is measured, so it has no bandwidth ports, but each column's link still passes its shim DMA."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array_strix.yaml")
    dram = accelerator.offchip_memory
    assert dram in accelerator.bandwidth and not [p for p in accelerator.ports if p.key.memory == "dram"]
    link = next(d["cl"] for u, v, d in accelerator.network.edges(data=True) if u is dram and v.name == "tile_3_1")
    port = dram.dma_port(link.ports[dram], READ, input_role(1))
    assert (port.key.port, port.channels) == ("mm2s_c3", 2)
    port = dram.dma_port(link.ports[dram], WRITE, OUTPUT)
    assert (port.key.port, port.channels) == ("s2mm_c3", 2)
    assert dram.dma_port(link.ports[dram], "read_by_datapath", OUTPUT) is None


def test_a_measured_memory_need_not_declare_its_cores_ports(tmp_path):
    """A memory with a measured bandwidth may declare no ports."""
    memories = _two_l1s()["memories"]
    memories["l1_a"]["ports"] = {}
    connections = _two_l1s()["connections"]
    for connection in connections:
        if "l1_a" in connection["ends"]:
            connection["ends"]["l1_a"] = []
    bandwidth = {"l1_a": {"ceiling": 64, "contiguous": 64}}
    accelerator = _accelerator(tmp_path, memories=memories, connections=connections, bandwidth=bandwidth)
    assert not [port for port in accelerator.ports if port.key.memory == "l1_a"]


def test_a_port_of_no_bandwidth_is_rejected():
    memories = _two_l1s()["memories"]
    memories["l3"]["ports"]["rw_port_1"]["bandwidth"] = 0
    assert any("`memories.l3.ports.rw_port_1.bandwidth` must be > 0" in e for e in _errors(memories=memories))


def test_a_view_of_two_memories_is_rejected_when_the_accelerator_is_built(tmp_path):
    """Building an accelerator from objects rejects a view shared by two memories."""
    accelerator = _accelerator(tmp_path)
    l1_a = accelerator.memories["l1_a"]
    twin = Memory("twin", l1_a.views, {}, offchip=False, capacity_bits=None)
    with pytest.raises(ValueError, match="0.sram_64KB is a view of both l1_a and twin"):
        Accelerator("two", accelerator.core_list, [*accelerator.memories.values(), twin], MemoryGraph())


def test_a_link_into_a_memory_no_core_views_sorts_in_the_steady_state_trace():
    """Links from DRAM, which has no views, still get an integer sort key in the trace."""
    accelerator = parse_accelerator("stream/inputs/examples/hardware/fusemax.yaml")
    dram = accelerator.offchip_memory
    links = [data["cl"] for u, _, data in accelerator.network.edges(data=True) if u is dram]
    assert links and all(isinstance(_resource_sort_key(link), int) for link in links)
    assert _resource_sort_key(links[0]) > _resource_sort_key(accelerator.get_core(0))


def test_a_port_naming_a_core_that_is_no_view_is_rejected():
    memories = _two_l1s()["memories"]
    memories["l1_a"]["ports"]["r_port_1"]["cores"] = [0, 1, 2]
    assert any("names cores [2], no views of it" in error for error in _errors(memories=memories))


@pytest.mark.parametrize(
    ("ports", "missing"),
    [
        ({"w_port_1": {"from": "w_port_1", "cores": [0, 1]}}, "core 0's port r_port_1 is in ports []"),
        (
            {
                "r_port_1": {"from": "r_port_1", "cores": [0, 1]},
                "r_again": {"from": "r_port_1", "cores": [1]},
                "w_port_1": {"from": "w_port_1", "cores": [0, 1]},
            },
            "core 1's port r_port_1 is in ports ['r_port_1', 'r_again']",
        ),
    ],
)
def test_every_port_a_view_declares_is_in_exactly_one_port_of_its_memory(tmp_path, ports, missing):
    data = _two_l1s()
    _declare_ports(data, "l1_a", ports)
    with pytest.raises(ValueError, match=re.escape(missing)):
        _accelerator(tmp_path, data)


def _declare_ports(data: dict, memory: str, ports: dict) -> None:
    """Give ``memory`` ``ports``, each connection reaching it through all of them."""
    data["memories"][memory]["ports"] = ports
    for connection in data["connections"]:
        if memory in connection["ends"]:
            connection["ends"][memory] = list(ports)


def _link(accelerator, sender: str, receiver: str, parallel: int = 0):
    """The link from memory ``sender`` to ``receiver``, the ``parallel``-th declared between them."""
    a, b = accelerator.memories[sender], accelerator.memories[receiver]
    (link,) = [
        d["cl"] for u, v, d in accelerator.network.edges(data=True) if (u, v) == (a, b) and d["cl"].parallel == parallel
    ]
    return link


def test_cores_of_one_memory_may_each_have_their_own_port(tmp_path):
    """Each core may have its own port; a link through several ports serving the same operand is rejected."""
    data = _two_l1s()
    _declare_ports(
        data,
        "l1_a",
        {
            f"{direction}_{core}": {"from": f"{direction}_port_1", "cores": [core]}
            for direction in ("r", "w")
            for core in (0, 1)
        },
    )
    with pytest.raises(ValueError, match=r"of memory l1_a, which all serve"):
        _accelerator(tmp_path, data)
    for connection in data["connections"]:
        if "l1_a" in connection["ends"]:
            connection["ends"]["l1_a"] = ["r_0", "w_0"]
    data["connections"].append(
        {"type": "link", "ends": {"l1_a": ["r_1", "w_1"], "l1_b": ["r_port_1", "w_port_1"]}, "bandwidth": 32}
    )
    accelerator = _accelerator(tmp_path, data)
    l1_a = [port for port in accelerator.ports if port.key.memory == "l1_a"]
    assert sorted((port.key.port, port.core_ids) for port in l1_a) == [
        ("r_0", (0,)),
        ("r_1", (1,)),
        ("w_0", (0,)),
        ("w_1", (1,)),
    ]
    assert [port.key.port for port in accelerator.ports.ports_of(accelerator.get_core(1))] == ["r_1", "w_1"]
    l1_a, l1_b = accelerator.memories["l1_a"], accelerator.memories["l1_b"]
    through_zero, through_one = _link(accelerator, "l1_a", "l1_b"), _link(accelerator, "l1_a", "l1_b", 1)

    def passed(memory, direction, links):
        return [port.key.port for port in accelerator.ports.memory_ports(memory, direction, OUTPUT, links)]

    assert passed(l1_a, READ, [through_zero]) == ["r_0"]
    assert passed(l1_a, READ, [through_one]) == ["r_1"]
    assert passed(l1_b, WRITE, [through_one]) == ["w_port_1"]
    assert passed(l1_a, READ, []) == []


def test_views_naming_their_ports_apart_leave_a_transfer_no_port(tmp_path):
    """A link through two ports that serve the same operand in the same direction is rejected."""
    core = yaml.safe_load(SHARED_CORE2.read_text())
    for port in core["memories"]["sram_64KB"]["ports"]:
        port["name"] = port["name"].replace("_1", "_2")
    data = _two_l1s()
    data["cores"][1] = core
    _declare_ports(
        data,
        "l1_a",
        {
            f"{direction}_port_{core}": {"from": f"{direction}_port_{core}", "cores": [core - 1]}
            for direction in ("r", "w")
            for core in (1, 2)
        },
    )
    with pytest.raises(ValueError, match=r"of memory l1_a, which all serve"):
        _accelerator(tmp_path, data)


class _Stub:
    """A node, tensor, core or memory with just the fields a test reads, hashable by identity."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_the_bits_a_position_moves_are_its_own_where_cores_share_a_memory():
    """Per-position bits of cores sharing a memory sum to the total, not to twice it."""
    a, b = _Stub(name="l1_a"), _Stub(name="l1_b")
    space = object.__new__(DecisionSpace)
    space.hardware = _Stub(shares_memory=lambda one, other: one is other)
    space.overlaps = lambda tr: {(0, 0): 3, (1, 0): 5, (1, 1): 7}  # type: ignore[method-assign]
    tr = _Stub(inputs=[_Stub(operand_type=_Stub(bitwidth=8))], outputs=[None])
    choice = _Stub(sources=(a, a), targets=(b, b))
    assert space.moved_bits(tr, choice) == 15 * 8
    assert [space.moved_bits(tr, choice, source_at=i) for i in range(2)] == [3 * 8, 12 * 8]
    assert [space.moved_bits(tr, choice, target_at=j) for j in range(2)] == [8 * 8, 7 * 8]


def test_a_core_reads_a_copy_in_one_memory():
    """Each reading core gets one memory for a copy; a core reading it from two memories is rejected."""
    a, b = _Stub(name="l1_a"), _Stub(name="l1_b")
    cores = [_Stub(id=i) for i in range(3)]
    one, other, tensor = _Stub(name="conv1"), _Stub(name="conv2"), _Stub(name="x")
    assert reader_positions(tensor, [(one, cores[:2], (a, a)), (other, cores[1:], (a, b))]) == (a, a, b)
    with pytest.raises(ValueError, match="core 1 reads x in l1_b for conv2 but in l1_a for conv1"):
        reader_positions(tensor, [(one, cores[:2], (a, a)), (other, cores[1:], (b, b))])


def test_fifo_depth_and_descriptors_count_a_tensor_once_per_position():
    """A tensor held by two cores in one memory counts its fifo and descriptors once per core."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "positions")
    model.set_param(SolverParams.VERBOSITY, 0)
    memory = Memory.__new__(Memory)
    core = _Stub(id=0, kind="compute", core_type="zigzag.compute")
    memory.name, memory.views, memory.offchip, memory.capacity_bits = "l1_a", ((core, "sram"),), False, 0
    memory.ports = {}
    tensor = _Stub(name="x")
    transfer = _Stub(name="Transfer(x)", inputs=[_Stub(name="x0")], outputs=[tensor])
    transfer.tensors = [*transfer.inputs, *transfer.outputs]
    space = object.__new__(DecisionSpace)
    space.transfer_nodes = (transfer,)
    space.tensor_choices = {tensor: ((memory, memory),), transfer.inputs[0]: ((_Stub(offchip=True),),)}
    space.candidate_memories = lambda t: set(space.tensor_choices[t][0])  # type: ignore[method-assign]
    space.is_fixed = lambda t: True  # type: ignore[method-assign]
    space.fixed_choice = lambda t: space.tensor_choices[t][0]  # type: ignore[method-assign]
    space.stops = lambda t: range(-1, 1)  # type: ignore[method-assign]
    space.tiles_needed_levels = {(tensor, -1): 1, (tensor, 0): 4}
    space.bds_needed_levels = {}
    z_stop = {(tensor, stop): model.add_var(vtype=SolverVarType.BINARY, name=f"z{stop}") for stop in (-1, 0)}
    model.add_constr(z_stop[(tensor, -1)] == 0)
    model.add_constr(z_stop[(tensor, 0)] == 1)
    q = QuantityRegistry()
    ctx = FormulationContext(space, DecisionVariables({}, {}, z_stop, {}, {}), model, q, ResourceLedger())
    ObjectFifoDepth().build(ctx)
    BufferDescriptors().build(ctx)
    depth, descriptors = q.get("object_fifo_depth", memory).expr, q.get("buffer_descriptor_depth", memory).expr
    model.set_objective(depth + descriptors)
    model.optimize()
    assert (model.value(depth), model.value(descriptors)) == (8, 8)
    assert ctx.ledger.terms[(OBJECT_FIFO_DEPTH, memory)] == {"x": 2}


SHIPPED_HARDWARE = sorted(
    str(path)
    for path in Path("stream/inputs").glob("**/hardware/*.yaml")
    if "cores" in (yaml.safe_load(path.read_text()) or {})
)


@pytest.mark.parametrize("path", SHIPPED_HARDWARE)
def test_every_core_a_hardware_file_declares_is_in_its_accelerator(path):
    declared = yaml.safe_load(Path(path).read_text())["cores"]
    assert sorted(core.id for core in parse_accelerator(path).core_list) == sorted(declared)


@pytest.mark.parametrize(("name", "columns"), [("whole_array", 4), ("whole_array_strix", 8)])
def test_aie_dram_is_an_off_chip_memory_with_a_shim_dma_per_column_and_every_mem_tile_reaches_it(name, columns):
    """AIE DRAM has no views, one shim DMA port pair per column, and a link to every memory tile through its shim."""
    accelerator = parse_accelerator(f"stream/inputs/aie/hardware/{name}.yaml")
    dram = accelerator.offchip_memory
    assert dram.views == () and {core.kind for core in accelerator.core_list} == {"compute", "memory"}
    assert {port.key.port: (port.bits_per_cycle, port.core_ids, port.channels) for port in dram.ports.values()} == {
        f"{direction}_c{k}": (128, (), 2) for k in range(columns) for direction in ("mm2s", "s2mm")
    }
    assert (dram.dma.buffer_descriptors, dram.dma.iterations) == (16, 64)
    for _, mem_tile, data in accelerator.network.edges(dram, data=True):
        assert data["cl"].ports[dram] == (f"mm2s_c{mem_tile.col_id}", f"s2mm_c{mem_tile.col_id}")
    mem_tiles = {memory for memory in accelerator.memories.values() if memory.kind == "memory"}
    assert {v for u, v in accelerator.network.edges() if u is dram} == mem_tiles


def _node(op_type: str, inputs: int) -> ComputationNode:
    tensors = [Tensor.create(f"t{i}", bf16, (8, 8)) for i in range(inputs + 1)]
    maps = tuple(AffineMap.from_callable(lambda m, n: (m, n)) for _ in tensors)
    return ComputationNode(
        type=op_type, name=op_type, inputs=tuple(tensors[:-1]), outputs=(tensors[-1],), operand_mapping=maps
    )


def test_a_multiply_accumulates_bias_lands_where_its_output_does():
    """A Gemm bias has the output role and lands in the activation SRAM on meta_prototype."""
    gemm = _node("Gemm", 3)
    assert [role_of(gemm, t) for t in (*gemm.inputs, *gemm.outputs)] == ["input1", "input2", OUTPUT, OUTPUT]
    accelerator = parse_accelerator(META_PROTOTYPE)
    core = accelerator.get_core(0)
    assert accelerator.memory_of(core, role_of(gemm, gemm.inputs[2])).name == "meta_prototype_0"


def test_every_input_of_an_elementwise_node_is_an_operand():
    add = _node("Add", 3)
    assert [role_of(add, t) for t in add.inputs] == ["input1", "input2", "input3"]


def test_views_declaring_a_shared_port_differently_are_rejected_when_parsed(tmp_path):
    """Views declaring a shared port differently fail at parse time."""
    core = yaml.safe_load(SHARED_CORE2.read_text())
    core["memories"]["sram_64KB"]["ports"][0]["bandwidth_max"] = 256
    data = _two_l1s()
    data["cores"][1] = core
    with pytest.raises(ValueError, match="declare r_port_1 differently"):
        _accelerator(tmp_path, data)


def test_carried_state_lands_where_the_output_does():
    """A linear-attention state update reads its carried state with the output role."""
    node = next(n for n in build_block("linear_attention").get_computation_nodes() if n.name == "state_update")
    assert [role_of(node, t) for t in node.inputs] == ["input1", "input2", OUTPUT]
