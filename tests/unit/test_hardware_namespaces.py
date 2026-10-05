"""MILP constraints attach by hardware namespace, discovered rather than hardcoded."""

from __future__ import annotations

import yaml

from stream.opt.allocation.constraint_optimization import hardware as ctx_module
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
    return NamespaceConfig(
        accelerator=accelerator,
        offchip_core_id=accelerator.offchip_core_id,
        mem_cores=(),
        nb_cols_to_use=4,
    )


def test_builtin_aie2_constraints_attach_through_the_plugin_path():
    """The built-in strategy is registered as an entry point, not special-cased in the builder."""
    accelerator = _accelerator(_AIE)
    strategies = build_hardware_facts(accelerator).namespaces
    assert [type(s).__name__ for s in strategies] == ["AIE2Namespace"]


def test_a_namespace_adds_its_families_to_the_default_set():
    """The AIE2 limits are families the namespace contributes, after Stream's own."""
    context = build_hardware_facts(_accelerator(_AIE))
    assert context.default_families == (*DEFAULT_FAMILIES, *AIE2Namespace.families)
    assert "aie2_dma_channels" in AIE2Namespace.families


def test_a_namespace_the_accelerator_lacks_contributes_nothing():
    context = build_hardware_facts(_accelerator(_ZIGZAG))
    assert context.namespaces == ()
    assert context.default_families == DEFAULT_FAMILIES


def test_an_overlay_namespace_is_picked_up(monkeypatch):
    """The point of the seam: proprietary hardware ships constraints without editing this file."""

    class AcmeConstraints(HardwareNamespace):
        NAMESPACE = "zigzag"  # stand in for a proprietary namespace present in the fixture

    monkeypatch.setattr(
        ctx_module,
        "load_group",
        lambda group: [LoadedPlugin("zigzag", AcmeConstraints, "vendor-overlay-acme", 20)],
    )
    accelerator = _accelerator(_ZIGZAG)
    strategies = namespaces_for(accelerator, _config(accelerator))
    assert [type(s).__name__ for s in strategies] == ["AcmeConstraints"]


def test_a_broken_strategy_is_skipped_not_raised(monkeypatch):
    class Exploding(HardwareNamespace):
        NAMESPACE = "zigzag"

        @classmethod
        def from_config(cls, config):
            raise RuntimeError("bad overlay")

    monkeypatch.setattr(
        ctx_module,
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
        ctx_module,
        "load_group",
        lambda group: [
            LoadedPlugin("zigzag", Baseline, "stream-dse", 0),
            LoadedPlugin("zigzag", Override, "vendor-overlay-acme", 20),
        ],
    )
    accelerator = _accelerator(_ZIGZAG)
    strategies = namespaces_for(accelerator, _config(accelerator))
    assert [type(s).__name__ for s in strategies] == ["Override"]


def test_from_config_maps_the_aie2_reconfiguration():
    accelerator = _accelerator(_AIE)
    built = AIE2Namespace.from_config(_config(accelerator))
    reconfiguration = accelerator.reconfiguration
    assert built.cycles_per_column == float(reconfiguration.get("cycles_per_column", 0.0))
    assert built.reset_cycles == float(reconfiguration.get("reset_cycles", 0.0))


def test_compute_tiles_reserve_the_toolchain_stack():
    accelerator = _accelerator(_AIE)
    context = build_hardware_facts(accelerator)
    compute = next(c for c in accelerator.core_list if c.type == "compute")
    memory = next(c for c in accelerator.core_list if c.type == "memory")
    assert context.reserved_memory_bits(compute) == AIE2Namespace.DEFAULT_CORE_STACK_BYTES * 8
    assert context.reserved_memory_bits(memory) == 0
