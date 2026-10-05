"""Tests for the solver's overlap evidence (``overlap_section``)."""

from __future__ import annotations

from types import SimpleNamespace

from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.report import overlap_section, resource_slack


def make_context(slack: dict[object, int], overlap: int | None, recurrence: int = 0) -> SimpleNamespace:
    """A solved formulation stub carrying only what the overlap section reads, so no solve is needed."""
    q = QuantityRegistry()
    for res, cycles in slack.items():
        q.add("idle_latency", cycles, index=res)
    if overlap is not None:
        q.add("overlap", overlap)
    q.add("recurrence_bound", recurrence)
    return SimpleNamespace(model=SimpleNamespace(value=float), quantities=q)


def core(core_id: int) -> Core:
    return Core(core_id=core_id, name=f"core_{core_id}", core_type="zigzag.compute")


class TestOverlapSection:
    def test_binding_is_the_argmin_when_overlap_sits_below_the_cap(self) -> None:
        """The binding set is the argmin of slack even when overlap sits below its cap (not equality)."""
        ctx = make_context({core(0): 14942, core(1): 181228, core(2): 14942}, overlap=14923)

        section = overlap_section(ctx, resource_slack(ctx))

        assert section["overlap_cycles"] == 14923
        assert section["binding_resources"] == [str(core(0)), str(core(2))]

    def test_binding_matches_an_equality_test_when_the_overlap_is_at_its_cap(self) -> None:
        """Where the old rule worked it must keep working: at the cap, argmin and equality agree."""
        ctx = make_context({core(0): 100, core(1): 250}, overlap=100)

        section = overlap_section(ctx, resource_slack(ctx))

        assert section["binding_resources"] == [str(core(0))]

    def test_zero_slack_pins_the_overlap_to_zero(self) -> None:
        """A resource busy from an early to a late slot has no boundary idle and binds at 0."""
        ctx = make_context({core(0): 0, core(1): 5000}, overlap=0)

        section = overlap_section(ctx, resource_slack(ctx))

        assert section["overlap_cycles"] == 0
        assert section["binding_resources"] == [str(core(0))]

    def test_no_resources_yields_no_binding_set(self) -> None:
        """Nothing to be binding is the one case where empty is the honest answer."""
        ctx = make_context({}, overlap=None)

        section = overlap_section(ctx, resource_slack(ctx))

        assert section["binding_resources"] == []
        assert section["per_resource_slack"] == []

    def test_recurrence_bound_is_carried_through(self) -> None:
        ctx = make_context({core(0): 100}, overlap=64, recurrence=64)

        assert overlap_section(ctx, resource_slack(ctx))["recurrence_bound_cycles"] == 64
