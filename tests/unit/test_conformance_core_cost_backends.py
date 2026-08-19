"""Conformance kit for the ``stream.core_cost_backends`` seam."""

from __future__ import annotations

import sys
import tempfile

from stream.api import optimize_allocation_co_generic
from stream.hardware.architecture.core import Core
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.plugins import LoadedPlugin
from stream.stages.estimation import core_cost_backends as backends_module
from stream.stages.estimation.core_cost_backends import (
    AIE_BACKEND,
    ZIGZAG_BACKEND,
    discover_backends,
    select_backend,
)

_AIE_ESTIMATOR_MODULE = "stream.stages.estimation.aie_cost_estimator"
_ZIGZAG_HARDWARE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"

_2CONV = TwoConvWorkloadConfig(
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


def _fake_core(core_type: str) -> Core:
    """A minimal core carrying only the fields a backend's ``claims`` reads."""
    core = Core.__new__(Core)
    core.id = 0
    core.core_type = core_type
    core.type = core_type.rsplit(".", maxsplit=1)[-1] if "." in core_type else core_type
    return core


# ---------------------------------------------------------------------------
# The contract every registered backend must satisfy
# ---------------------------------------------------------------------------


def test_registered_backends_declare_the_contract() -> None:
    discovered = discover_backends()
    assert {b.name for b in discovered} >= {"aie", "zigzag"}, "built-in backends must register as entry points"
    for backend in discovered:
        assert isinstance(backend.name, str) and backend.name, f"{backend!r} must advertise a str name"
        assert isinstance(backend.priority, int), f"{backend.name!r} must advertise an int priority"
        assert callable(backend.claims), f"{backend.name!r}.claims must be callable"
        assert callable(backend.make), f"{backend.name!r}.make must be callable"


def test_claims_returns_bool_for_every_backend() -> None:
    for backend in discover_backends():
        for core_type in ("aie2.compute", "aie2.memory", "zigzag.compute", "compute"):
            verdict = backend.claims(_fake_core(core_type))
            assert isinstance(verdict, bool), f"{backend.name!r}.claims must return a bool, got {verdict!r}"


def test_claims_is_cheap_and_imports_nothing_heavy() -> None:
    """``claims`` must decide without importing the toolchain -- no AIE estimator import when unselected."""
    sys.modules.pop(_AIE_ESTIMATOR_MODULE, None)
    assert AIE_BACKEND.claims(_fake_core("aie2.compute")) is True
    assert AIE_BACKEND.claims(_fake_core("compute")) is False
    assert _AIE_ESTIMATOR_MODULE not in sys.modules, "claims() must not trigger the lazy AIE import"


# ---------------------------------------------------------------------------
# Dispatch: the built-in default reproduces the old is_aie_compute_core branch
# ---------------------------------------------------------------------------


def test_aie_compute_core_selects_the_aie_backend() -> None:
    assert select_backend(_fake_core("aie2.compute")).name == "aie"


def test_non_aie_cores_fall_back_to_zigzag() -> None:
    for core_type in ("aie2.memory", "aie2.shim", "zigzag.compute", "compute"):
        assert select_backend(_fake_core(core_type)).name == "zigzag", core_type


def test_zigzag_is_the_universal_claimant() -> None:
    for core_type in ("aie2.compute", "aie2.memory", "anything", "compute"):
        assert ZIGZAG_BACKEND.claims(_fake_core(core_type)) is True


# ---------------------------------------------------------------------------
# The seam's purpose: an overlay backend for its own namespace is selected
# ---------------------------------------------------------------------------


class _FakeOverlayBackend:
    """Stands in for a customer overlay's backend for a proprietary hardware namespace."""

    name = "acme_npu"
    priority = 50  # above zigzag's 0

    def claims(self, core: Core) -> bool:
        return str(core.core_type).startswith("acme.")

    def make(self, context):  # pragma: no cover - selection is what is under test here
        raise NotImplementedError


def test_overlay_backend_above_zigzag_is_selected_and_tags_through(monkeypatch) -> None:
    monkeypatch.setattr(
        backends_module,
        "load_group",
        lambda group: [
            LoadedPlugin("zigzag", ZIGZAG_BACKEND, "stream-dse", 0),
            LoadedPlugin("aie", AIE_BACKEND, "stream-dse", 0),
            LoadedPlugin("acme_npu", _FakeOverlayBackend(), "vendor-overlay-acme", 20),
        ],
    )
    # The overlay claims its own namespace and outranks the universal zigzag fallback there.
    assert select_backend(_fake_core("acme.compute")).name == "acme_npu"
    # It does not disturb the built-in decisions for cores it does not claim.
    assert select_backend(_fake_core("aie2.compute")).name == "aie"
    assert select_backend(_fake_core("zigzag.compute")).name == "zigzag"


def test_higher_priority_wins_when_two_backends_claim_the_same_core(monkeypatch) -> None:
    class _Loud(_FakeOverlayBackend):
        name = "loud"
        priority = 99

        def claims(self, core: Core) -> bool:  # noqa: ARG002
            return True

    monkeypatch.setattr(
        backends_module,
        "load_group",
        lambda group: [
            LoadedPlugin("zigzag", ZIGZAG_BACKEND, "stream-dse", 0),
            LoadedPlugin("loud", _Loud(), "vendor-overlay-acme", 20),
        ],
    )
    assert select_backend(_fake_core("compute")).name == "loud"


# ---------------------------------------------------------------------------
# A produced CoreCostEntry is well-formed (real backend, existing fixture)
# ---------------------------------------------------------------------------


def test_produced_entries_carry_ideal_cycle_and_a_backend_tag() -> None:
    """Drive the real ZigZag backend end-to-end and check the fields the inspection contract reads."""
    workload_path = make_2_conv_workload(_2CONV)
    registered_names = {b.name for b in discover_backends()} | {"ideal-cycle"}
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = optimize_allocation_co_generic(
            hardware=_ZIGZAG_HARDWARE,
            workload=workload_path,
            experiment_id="conformance_core_cost_backends",
            output_path=tmpdir,
        )
        cost_lut = ctx.get("cost_lut")
        entries = [entry for core_dict in cost_lut.lut.values() for entry in core_dict.values()]
        assert entries, "expected the CO run to populate the cost LUT"
        for entry in entries:
            assert entry.ideal_cycle is not None
            assert float(entry.ideal_cycle) >= 0
            tag = entry.metadata.get("backend")
            assert tag in registered_names, f"backend tag {tag!r} is not a registered backend name"
