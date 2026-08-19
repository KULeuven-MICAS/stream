"""MILP constraints attach by hardware namespace, discovered rather than hardcoded."""

from __future__ import annotations

import yaml

from stream.opt.allocation.constraint_optimization import context as ctx_module
from stream.opt.allocation.constraint_optimization.context import (
    AIE2Constraints,
    NamespaceConstraintConfig,
    NamespaceConstraints,
    build_transfer_context,
    namespace_constraints_for,
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


def _config(accelerator) -> NamespaceConstraintConfig:
    return NamespaceConstraintConfig(
        accelerator=accelerator,
        offchip_core_id=accelerator.offchip_core_id,
        mem_cores=(),
        nb_cols_to_use=4,
        max_compute_tile_dma_channels=8,
        max_mem_tile_dma_channels=6,
        max_shim_tile_dma_channels=2,
    )


def test_builtin_aie2_constraints_attach_through_the_plugin_path():
    """The built-in strategy is registered as an entry point, not special-cased in the builder."""
    accelerator = _accelerator(_AIE)
    strategies = build_transfer_context(accelerator).namespace_constraints
    assert [type(s).__name__ for s in strategies] == ["AIE2Constraints"]
    assert strategies[0].max_mem_tile_dma_channels == 6


def test_builder_arguments_reach_the_strategy():
    accelerator = _accelerator(_AIE)
    strategies = build_transfer_context(accelerator, max_mem_tile_dma_channels=3).namespace_constraints
    assert strategies[0].max_mem_tile_dma_channels == 3


def test_a_namespace_the_accelerator_lacks_contributes_nothing():
    accelerator = _accelerator(_ZIGZAG)
    assert build_transfer_context(accelerator).namespace_constraints == ()


def test_an_overlay_namespace_is_picked_up(monkeypatch):
    """The point of the seam: proprietary hardware ships constraints without editing this file."""

    class AcmeConstraints(NamespaceConstraints):
        NAMESPACE = "zigzag"  # stand in for a proprietary namespace present in the fixture

    monkeypatch.setattr(
        ctx_module,
        "load_group",
        lambda group: [LoadedPlugin("zigzag", AcmeConstraints, "vendor-overlay-acme", 20)],
    )
    accelerator = _accelerator(_ZIGZAG)
    strategies = namespace_constraints_for(accelerator, _config(accelerator))
    assert [type(s).__name__ for s in strategies] == ["AcmeConstraints"]


def test_a_broken_strategy_is_skipped_not_raised(monkeypatch):
    class Exploding(NamespaceConstraints):
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
    assert namespace_constraints_for(accelerator, _config(accelerator)) == []


def test_highest_priority_registration_wins(monkeypatch):
    """load_group returns lowest priority first; the last registration for a namespace is kept."""

    class Baseline(NamespaceConstraints):
        NAMESPACE = "zigzag"

    class Override(NamespaceConstraints):
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
    strategies = namespace_constraints_for(accelerator, _config(accelerator))
    assert [type(s).__name__ for s in strategies] == ["Override"]


def test_from_config_maps_the_aie2_knobs():
    accelerator = _accelerator(_AIE)
    built = AIE2Constraints.from_config(_config(accelerator))
    assert built.max_compute_tile_dma_channels == 8
    assert built.max_shim_tile_dma_channels == 2
    assert built.offchip_core_id == accelerator.offchip_core_id
