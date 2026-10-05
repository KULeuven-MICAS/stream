"""Allocation facts and families attach by hardware namespace, discovered rather than hardcoded."""

from __future__ import annotations

import pytest
import yaml

from stream.api import default_families
from stream.opt.allocation.constraint_optimization import hardware as hardware_module
from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES
from stream.opt.allocation.constraint_optimization.hardware import (
    AIE2Namespace,
    HardwareNamespace,
    NamespaceConfig,
    build_hardware_facts,
    namespaces_for,
)
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.plugins import LoadedPlugin

_AIE = "stream/inputs/aie/hardware/whole_array_strix.yaml"
_ZIGZAG = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def _accelerator(path: str):
    data = yaml.safe_load(open(path))
    validator = AcceleratorValidator(data, path)
    data, _ = validator.normalized_data, validator.validate()
    return AcceleratorFactory(data).create()


def _config(accelerator) -> NamespaceConfig:
    return NamespaceConfig(accelerator)


def test_the_builtin_aie2_namespace_attaches_without_an_entry_point(monkeypatch):
    """Stream's own namespace is registered in-tree, so an install whose entry points are stale still has it."""
    monkeypatch.setattr(hardware_module, "load_group", lambda group: [])
    namespaces = build_hardware_facts(_accelerator(_AIE), 4).namespaces
    assert [type(s).__name__ for s in namespaces] == ["AIE2Namespace"]


def test_a_namespace_adds_its_families_to_the_default_set():
    """The AIE2 limits are families the namespace contributes, after Stream's own."""
    assert default_families(_accelerator(_AIE)) == (*DEFAULT_FAMILIES, *AIE2Namespace.families)
    assert "aie2_dma_channels" in AIE2Namespace.families


def test_a_namespace_the_accelerator_lacks_contributes_nothing():
    accelerator = _accelerator(_ZIGZAG)
    assert build_hardware_facts(accelerator, 4).namespaces == ()
    assert default_families(accelerator) == DEFAULT_FAMILIES


@pytest.mark.parametrize("hook", ["add_object_fifo_constraints", "add_dma_usage_constraints"])
def test_a_namespace_with_a_removed_hook_is_rejected(monkeypatch, hook):
    """A 1.x namespace that still constrains the model through a hook is told which family replaces it."""

    class Legacy(HardwareNamespace):
        NAMESPACE = "zigzag"

    setattr(Legacy, hook, lambda self, *args: None)
    monkeypatch.setattr(hardware_module, "load_group", lambda group: [LoadedPlugin("zigzag", Legacy, "old", 0)])
    accelerator = _accelerator(_ZIGZAG)
    with pytest.raises(TypeError, match=rf"{hook}.*aie2_"):
        namespaces_for(accelerator, _config(accelerator))


def test_an_overlay_namespace_is_picked_up(monkeypatch):
    """The point of the seam: proprietary hardware ships its facts and families without editing this file."""

    class AcmeNamespace(HardwareNamespace):
        NAMESPACE = "zigzag"

    monkeypatch.setattr(
        hardware_module,
        "load_group",
        lambda group: [LoadedPlugin("zigzag", AcmeNamespace, "vendor-overlay-acme", 20)],
    )
    accelerator = _accelerator(_ZIGZAG)
    namespaces = namespaces_for(accelerator, _config(accelerator))
    assert [type(s).__name__ for s in namespaces] == ["AcmeNamespace"]


def test_a_broken_namespace_is_skipped_not_raised(monkeypatch):
    class Exploding(HardwareNamespace):
        NAMESPACE = "zigzag"

        @classmethod
        def from_config(cls, config):
            raise RuntimeError("bad overlay")

    monkeypatch.setattr(
        hardware_module,
        "load_group",
        lambda group: [LoadedPlugin("zigzag", Exploding, "vendor-overlay-broken", 20)],
    )
    accelerator = _accelerator(_ZIGZAG)
    assert namespaces_for(accelerator, _config(accelerator)) == []


def test_highest_priority_registration_wins(monkeypatch):
    """load_group returns lowest priority first; the last registration for a namespace is kept."""

    class Baseline(HardwareNamespace):
        NAMESPACE = "zigzag"

    class Override(HardwareNamespace):
        NAMESPACE = "zigzag"

    monkeypatch.setattr(
        hardware_module,
        "load_group",
        lambda group: [
            LoadedPlugin("zigzag", Baseline, "stream-dse", 0),
            LoadedPlugin("zigzag", Override, "vendor-overlay-acme", 20),
        ],
    )
    accelerator = _accelerator(_ZIGZAG)
    namespaces = namespaces_for(accelerator, _config(accelerator))
    assert [type(s).__name__ for s in namespaces] == ["Override"]


def test_from_config_maps_the_aie2_reconfiguration():
    accelerator = _accelerator(_AIE)
    built = AIE2Namespace.from_config(_config(accelerator))
    reconfiguration = accelerator.reconfiguration
    assert built.cycles_per_column == float(reconfiguration.get("cycles_per_column", 0.0))
    assert built.reset_cycles == float(reconfiguration.get("reset_cycles", 0.0))


def test_compute_tiles_reserve_the_toolchain_stack():
    accelerator = _accelerator(_AIE)
    context = build_hardware_facts(accelerator, 4)
    compute = next(c for c in accelerator.core_list if c.type == "compute")
    memory = next(c for c in accelerator.core_list if c.type == "memory")
    assert context.reserved_memory_bits(compute) == AIE2Namespace.DEFAULT_CORE_STACK_BYTES * 8
    assert context.reserved_memory_bits(memory) == 0


def test_an_entry_point_in_the_removed_group_is_ignored_with_a_warning(monkeypatch, caplog):
    """A 1.x install that registers its namespace in ``stream.constraints`` is told where it belongs now."""
    plugins = {"stream.constraints": [LoadedPlugin("zigzag", HardwareNamespace, "old-overlay", 0)]}
    monkeypatch.setattr(hardware_module, "load_group", lambda group: plugins.get(group, []))
    hardware_module._warn_removed_group.cache_clear()
    try:
        with caplog.at_level("WARNING"):
            assert hardware_module.namespace_classes(_accelerator(_ZIGZAG)) == {}
    finally:
        hardware_module._warn_removed_group.cache_clear()
    assert "'old-overlay' registers namespace 'zigzag' in the 'stream.constraints'" in caplog.text
