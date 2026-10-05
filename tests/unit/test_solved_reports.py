"""A report that cannot be computed is logged and left None, never failing the solve it reports on."""

import logging
from types import SimpleNamespace

import pytest

from stream.opt.allocation.constraint_optimization import report


class _BrokenReport:
    name = "broken"

    def report(self, ctx):
        raise ZeroDivisionError("report bug")


class _Report:
    name = "fine"

    def report(self, ctx):
        return {"fine": [1]}


def test_a_family_report_that_fails_leaves_its_section_none(monkeypatch: pytest.MonkeyPatch, caplog):
    monkeypatch.setattr(report, "_utilization", lambda *args: {"per_node": {}})
    monkeypatch.setattr(report, "overlap_section", lambda *args: {})
    families = SimpleNamespace(families=[_BrokenReport(), _Report()])
    with caplog.at_level(logging.WARNING):
        performance = report._performance(None, families, None, 0, [], [], [])
    assert performance["broken"] is None
    assert performance["fine"] == [1]
    assert "broken report" in caplog.text


def test_each_report_is_guarded_on_its_own(monkeypatch: pytest.MonkeyPatch, caplog):
    """As on 1.x, a failing performance report is None while the capacity slack and the breakdown are still read."""

    def fail(*args):
        raise KeyError("missing quantity")

    monkeypatch.setattr(report, "tensor_reuse_breakdown", lambda *args: [])
    monkeypatch.setattr(report, "resource_slack", lambda *args: [])
    monkeypatch.setattr(report, "memory_occupancy", lambda *args: [])
    monkeypatch.setattr(report, "_performance", fail)
    monkeypatch.setattr(report, "capacity_slack", lambda *args: {0: {"memory_bytes": 1.0}})
    monkeypatch.setattr(report, "slot_latency_breakdown", lambda *args: {"slots": []})
    with caplog.at_level(logging.WARNING):
        performance, slack, breakdown = report.solved_reports(None, None, {}, 0, None)
    assert performance is None
    assert (slack, breakdown) == ({0: {"memory_bytes": 1.0}}, {"slots": []})
    assert "performance report" in caplog.text
