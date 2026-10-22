"""Memories apart from cores: several cores compute out of one memory, which none of them owns."""

from __future__ import annotations

import pytest
import yaml

from stream.cost_model.communication_manager import DemandItem
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.parsing.accelerator_parser import parse_accelerator

TWO_L1S = """
name: two-l1s
cores:
  0: shared_testing_core1.yaml
  1: shared_testing_core2.yaml
  2: shared_testing_core1.yaml
  3: shared_testing_core2.yaml
  4: offchip.yaml
offchip_core_id: 4
unit_energy_cost: 0
memories:
  l1_a: [0.sram_64KB, 1.sram_64KB]
  l1_b: [2.sram_64KB, 3.sram_64KB]
  l3: [4.dram]
connections:
  - {type: link, memories: [l1_a, l3], bandwidth: 64}
  - {type: link, memories: [l1_b, l3], bandwidth: 64}
  - {type: link, memories: [l1_a, l1_b], bandwidth: 32}
"""
"""Two L1s of two cores each, joined to each other and to L3."""

WHOLE = ((0, 15), (0, 15))


def _accelerator(tmp_path, text: str = TWO_L1S, **overrides):
    data = yaml.safe_load(text) | overrides
    path = tmp_path / "hw.yaml"
    path.write_text(yaml.safe_dump(data))
    return parse_accelerator(str(path))


def _errors(tmp_path, **overrides) -> list[str]:
    data = yaml.safe_load(TWO_L1S) | overrides
    (tmp_path / "hw.yaml").write_text(yaml.safe_dump(data))
    validator = AcceleratorValidator(data, str(tmp_path / "hw.yaml"))
    assert not validator.validate()
    return validator.errors


def test_the_cores_of_a_memory_share_it_and_none_owns_it(tmp_path):
    accelerator = _accelerator(tmp_path)
    a = accelerator.memories["l1_a"]
    assert [(core.id, instance) for core, instance in a.views] == [(0, "sram_64KB"), (1, "sram_64KB")]
    assert accelerator.memory_of(accelerator.get_core(0)) is accelerator.memory_of(accelerator.get_core(1)) is a
    assert accelerator.offchip_memory is accelerator.memories["l3"] and accelerator.offchip_memory.offchip
    assert set(accelerator.network.nodes()) == set(accelerator.memories.values())


def test_any_core_of_a_memory_sends_over_its_links(tmp_path):
    """Either core of one L1 reaches either core of the other over the link between the two L1s."""
    accelerator = _accelerator(tmp_path)
    for source, target in ((0, 2), (1, 3), (1, 2)):
        plan = accelerator.communication_manager.route(
            (accelerator.get_core(source),),
            (accelerator.get_core(target),),
            (DemandItem(0, (0,), WHOLE),),
            16,
            lambda one, other: False,
        )
        assert [str(link) for link in plan.links_used] == ["CL(l1_a, l1_b, bw=32)"]


def test_a_bandwidth_model_is_keyed_by_its_memory(tmp_path):
    accelerator = _accelerator(tmp_path, bandwidth={"l1_a": {"ceiling": 64, "contiguous": 64}})
    assert list(accelerator.bandwidth) == [accelerator.memories["l1_a"]]


@pytest.mark.parametrize("key", [0, "core0", "0.rw_port_1", "l1_a.sram_64KB.rw_port_1"])
def test_a_bandwidth_key_naming_no_memory_is_rejected(tmp_path, key):
    assert _errors(tmp_path, bandwidth={key: {"ceiling": 64, "contiguous": 64}})


def test_a_port_key_of_a_measured_memory_is_rejected(tmp_path):
    model = {"ceiling": 64, "contiguous": 64}
    errors = _errors(tmp_path, bandwidth={"l1_a": model, "l1_a.rw_port_1": model})
    assert any("already has a measured bandwidth" in error for error in errors)


def test_a_core_in_two_memories_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["1.sram_64KB", "2.sram_64KB", "3.sram_64KB"]}
    errors = _errors(tmp_path, memories=views | {"l3": ["4.dram"]})
    assert any("core 1 must be a view of exactly one memory" in error for error in errors)


def test_a_core_in_no_memory_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["2.sram_64KB"], "l3": ["4.dram"]}
    errors = _errors(tmp_path, memories=views)
    assert any("core 3 must be a view of exactly one memory" in error for error in errors)


def test_a_memory_no_connection_reaches_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.sram_64KB"], "l1_b": ["2.sram_64KB"], "spare": ["3.sram_64KB"]}
    errors = _errors(tmp_path, memories=views | {"l3": ["4.dram"]})
    assert any("spare" in error and "no connection" in error for error in errors)


def test_a_link_from_a_memory_to_itself_is_rejected(tmp_path):
    links = [{"type": "link", "memories": ["l1_a", "l1_a"], "bandwidth": 64}]
    assert any("joins a memory to itself" in error for error in _errors(tmp_path, connections=links))


def test_a_connection_naming_cores_is_rejected(tmp_path):
    links = [{"type": "link", "cores": [0, 2], "bandwidth": 64}]
    assert _errors(tmp_path, connections=links)


def test_a_view_of_an_instance_its_core_does_not_declare_is_rejected(tmp_path):
    views = {"l1_a": ["0.sram_64KB", "1.l1"], "l1_b": ["2.sram_64KB", "3.sram_64KB"], "l3": ["4.dram"]}
    assert any("core 1 declares no memory `l1`" in error for error in _errors(tmp_path, memories=views))


def test_a_view_of_an_inner_level_is_rejected(tmp_path):
    """A view binds a core's top-level instance: an inner level is the core's own, not where transfers land."""
    text = """
name: inner
cores: {0: testing_core1.yaml, 1: offchip.yaml}
offchip_core_id: 1
memories: {l1: [0.sram_64KB], l3: [1.dram]}
connections: [{type: link, memories: [l1, l3], bandwidth: 64}]
"""
    with pytest.raises(ValueError, match="is no top-level memory"):
        _accelerator(tmp_path, text)
