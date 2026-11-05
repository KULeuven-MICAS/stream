"""Choosing a core-cost backend by name, giving backends options, and reusing cached entries only from the same
backend with the same cache identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND, CoreCostEntry
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.families import traffic
from stream.opt.allocation.constraint_optimization.layouts import CopyLayouts
from stream.plugins import LoadedPlugin
from stream.stages.estimation import core_cost_backends as backends_module
from stream.stages.estimation.core_cost_backends import (
    AIE_BACKEND,
    COST_BACKEND,
    ZIGZAG_BACKEND,
    entry_backend,
    select_backend,
)
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage

from .cost_stage_runs import priced


def _fake_core(core_type: str) -> Core:
    core = Core.__new__(Core)
    core.id = 0
    core.core_type = core_type
    core.type = core_type.rsplit(".", maxsplit=1)[-1]
    return core


class _Counting:
    """A backend that claims ZigZag compute cores, prices every node at one cycle and counts what it prices."""

    priority = -5

    def __init__(self, name: str = "counting") -> None:
        self.name = name
        self.priced: list[tuple[str, int]] = []

    def claims(self, core: Core) -> bool:
        return core.core_type == "zigzag.compute"

    def make(self, context: Any) -> _Counting:
        return self

    def estimate(self, node: Any, core: Core) -> CoreCostEntry:
        self.priced.append((node.name, core.id))
        return CoreCostEntry(1.0, 1.0, 1.0, 1.0, layer=node, metadata={"backend": self.name})


class _Configurable(_Counting):
    """A counting backend that takes one option, ``level``, which decides its entries."""

    def cache_identity(self, options: Mapping[str, Any]) -> dict[str, Any]:
        if unknown := set(options) - {"level"}:
            raise ValueError(f"unknown option {sorted(unknown)[0]}")
        return {"level": options.get("level", 0)}


def _registered(monkeypatch: pytest.MonkeyPatch, *extra: Any) -> None:
    plugins = [
        LoadedPlugin("zigzag", ZIGZAG_BACKEND, "stream-dse", 0),
        LoadedPlugin("aie", AIE_BACKEND, "stream-dse", 0),
    ]
    plugins += [LoadedPlugin(backend.name, backend, "test", 0) for backend in extra]
    monkeypatch.setattr(backends_module, "load_group", lambda group: plugins)


def test_a_preferred_backend_costs_the_cores_it_claims_and_the_priority_rule_the_rest(monkeypatch):
    counting = _Counting()
    _registered(monkeypatch, counting)
    assert select_backend(_fake_core("zigzag.compute"), prefer="counting") is counting
    assert select_backend(_fake_core("aie2.compute"), prefer="counting") is AIE_BACKEND
    assert select_backend(_fake_core("zigzag.compute")) is ZIGZAG_BACKEND


def test_preferring_a_backend_no_one_registered_is_refused(monkeypatch):
    _registered(monkeypatch)
    with pytest.raises(ValueError, match="unknown core-cost backend 'nope'; registered: aie, zigzag"):
        select_backend(_fake_core("zigzag.compute"), prefer="nope")


@pytest.mark.parametrize(
    "options, refusal",
    [
        ({"other": {}}, r"unknown core-cost backends \['other'\]"),
        ({"aie": {"level": 1}}, "the 'aie' core-cost backend takes no options"),
        ({"zigzag": {"level": 1}}, r"core_cost_options\['zigzag'\]: unknown options \['level'\]"),
        ({"counting": {"colour": "red"}}, "unknown option colour"),
    ],
)
def test_a_run_refuses_options_for_an_unknown_backend_or_one_that_does_not_take_them(
    monkeypatch, tmp_path, options, refusal
):
    _registered(monkeypatch, _Configurable())
    with pytest.raises(ValueError, match=refusal):
        priced(tmp_path, core_cost_options=options)


def test_an_entry_is_attributed_to_the_backend_that_priced_it(monkeypatch):
    counting = _Counting()
    _registered(monkeypatch, counting)
    core = _fake_core("zigzag.compute")

    def entry(metadata: dict[str, Any]) -> CoreCostEntry:
        return CoreCostEntry(1.0, 1.0, 1.0, 1.0, metadata=metadata)

    assert entry_backend(entry({"backend": "counting"}), core) is counting
    assert entry_backend(entry({"backend": IDEAL_CYCLE_BACKEND}), core) is ZIGZAG_BACKEND
    assert entry_backend(entry({}), core) is ZIGZAG_BACKEND
    assert entry_backend(entry({"backend": "uninstalled"}), core) is None


def _tags(stage: CoreCostEstimationStage) -> set[str]:
    return {entry.metadata["backend"] for per_core in stage.cost_lut.lut.values() for entry in per_core.values()}


def test_a_run_preferring_a_backend_reprices_entries_another_backend_cached(monkeypatch, tmp_path):
    default, preferred = _Counting("default"), _Counting("preferred")
    default.priority = 5
    _registered(monkeypatch, default, preferred)
    assert _tags(priced(tmp_path)) == {"default"}
    assert _tags(priced(tmp_path, core_cost_backend="preferred")) == {"preferred"}
    assert len(preferred.priced) == 2, "one entry per distinct node; the equal cores share it"
    assert _tags(priced(tmp_path)) == {"default"}
    assert len(default.priced) == 4


def test_a_cached_entry_is_reused_only_with_the_same_cache_identity(monkeypatch, tmp_path):
    configurable = _Configurable()
    _registered(monkeypatch, configurable)
    first = priced(tmp_path, core_cost_backend="counting", core_cost_options={"counting": {"level": 1}})
    entries = [entry for per_core in first.cost_lut.lut.values() for entry in per_core.values()]
    assert all(entry.metadata[COST_BACKEND] == ("counting", {"level": 1}) for entry in entries)
    once = len(configurable.priced)
    priced(tmp_path, core_cost_backend="counting", core_cost_options={"counting": {"level": 1}})
    assert len(configurable.priced) == once
    priced(tmp_path, core_cost_backend="counting", core_cost_options={"counting": {"level": 2}})
    assert len(configurable.priced) == 2 * once


def test_a_run_naming_a_backend_that_claims_none_of_its_cores_is_refused(monkeypatch, tmp_path):
    aie_only = _Counting("aie_only")
    aie_only.claims = lambda core: core.core_type.startswith("aie2.")
    _registered(monkeypatch, aie_only)
    with pytest.raises(ValueError, match="'aie_only' claims none of the cores this run costs"):
        priced(tmp_path, core_cost_backend="aie_only")


class _Describing(_Counting):
    """A backend whose entries say what they move through the core's ports and read together."""

    def port_traffic(self, entry: CoreCostEntry) -> tuple[tuple[str, str, float], ...]:
        return (("input1", "read_by_datapath", 64.0),)

    def contiguous_axes(self, entry: CoreCostEntry) -> tuple[frozenset[int], ...]:
        return (frozenset({1}), frozenset({0}), frozenset({1}))


@dataclass(frozen=True)
class _Node:
    name: str
    tensors: tuple[str, ...]


def _space_with_one_entry(backend_name: str) -> tuple[Any, Any, Any]:
    """A decision space holding one node on one ZigZag compute core, priced by ``backend_name``."""
    node = _Node("mm", ("A", "B", "C"))
    core = _fake_core("zigzag.compute")
    entry = CoreCostEntry(1.0, 1.0, 1.0, 1.0, layer=node, metadata={"backend": backend_name})
    cost_lut = SimpleNamespace(get_cores=lambda n: [core], get_cost=lambda n, c: entry)
    ports = SimpleNamespace(port_for=lambda core, direction, operand, memory: (direction, operand))
    accelerator = SimpleNamespace(ports=ports, memory_holding=lambda core, operand: "sram")
    space = SimpleNamespace(
        ssc_nodes=[node],
        ssis=None,
        slot_of={node: 0},
        cost_lut=cost_lut,
        accelerator=accelerator,
        mapping=SimpleNamespace(get=lambda n: SimpleNamespace(resource_allocation=[[core]])),
        workload=SimpleNamespace(nodes=lambda: [node]),
    )
    return space, node, core


def test_the_allocation_reads_port_traffic_from_the_backend_that_priced_each_entry(monkeypatch):
    _registered(monkeypatch, _Describing("describing"))
    monkeypatch.setattr(traffic, "active_fraction", lambda node, ssis: 1.0)
    space, _, _ = _space_with_one_entry("describing")
    moved = traffic.node_traffic(SimpleNamespace(space=space))
    assert [(t.port, t.bits) for t in moved] == [(("read_by_datapath", "input1"), 64.0)]


def test_the_allocation_reads_operand_layouts_from_the_backend_that_priced_each_entry(monkeypatch):
    _registered(monkeypatch, _Describing("describing"))
    space, node, _ = _space_with_one_entry("describing")
    assert CopyLayouts(space).needs(node, "B") == frozenset({0})


class _Delegating(_Counting):
    """A backend that has ZigZag price what it cannot, its entries tagged by the backend that priced them."""

    def estimate(self, node: Any, core: Core) -> CoreCostEntry:
        self.priced.append((node.name, core.id))
        return CoreCostEntry(1.0, 1.0, 1.0, 1.0, layer=node, metadata={"backend": "zigzag"})


def test_an_entry_another_backend_priced_for_the_chosen_one_is_reused_by_the_next_such_run(monkeypatch, tmp_path):
    delegating = _Delegating("delegating")
    _registered(monkeypatch, delegating)
    first = priced(tmp_path, core_cost_backend="delegating")
    entries = [entry for per_core in first.cost_lut.lut.values() for entry in per_core.values()]
    assert {(e.metadata["backend"], e.metadata[COST_BACKEND]) for e in entries} == {("zigzag", ("delegating", None))}
    once = len(delegating.priced)
    priced(tmp_path, core_cost_backend="delegating")
    assert len(delegating.priced) == once
