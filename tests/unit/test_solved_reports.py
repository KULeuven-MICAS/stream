"""A report that cannot be computed is logged and left None, never failing the solve it reports on."""

import logging
from typing import Any

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import families, report
from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


class _BrokenReport:
    name = "broken_report"
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def build(self, ctx: Any) -> None: ...

    def report(self, ctx: Any) -> dict[str, Any]:
        raise ZeroDivisionError("report bug")


def _no_occupancy(ctx: Any) -> list[dict[str, Any]]:
    raise KeyError("no occupancy")


@pytest.fixture(scope="module")
def solved(tmp_path_factory: pytest.TempPathFactory, two_conv: TwoConvWorkloadConfig) -> tuple[Any, str]:
    """A solve whose broken_report family and memory occupancy both fail to report, and what it logged."""
    known = {**families.available_families(), _BrokenReport.name: _BrokenReport}
    options = SolveOptions(families=[*DEFAULT_FAMILIES, _BrokenReport.name], artifacts=False)
    workload, mapping = make_2_conv_workload(two_conv), make_2_conv_mapping(two_conv)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(families, "available_families", lambda: known)
        patch.setattr(report, "memory_occupancy", _no_occupancy)
        handler = _Records()
        logging.getLogger(report.__name__).addHandler(handler)
        try:
            estimate = evaluate_mapping(
                ACCELERATOR, workload, str(tmp_path_factory.mktemp("reports")), mapping, options
            )
        finally:
            logging.getLogger(report.__name__).removeHandler(handler)
    return estimate.context.get("allocation").solution, "\n".join(handler.messages)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_a_family_report_that_fails_leaves_its_section_none(solved: tuple[Any, str]) -> None:
    solution, logged = solved
    assert solution.performance["broken_report"] is None
    assert "broken_report report" in logged


def test_each_report_is_guarded_on_its_own(solved: tuple[Any, str]) -> None:
    """The memory occupancy failing leaves it None and the capacity slack without memory, while the rest of the
    performance report and the slot latency breakdown are still read off the solution."""
    solution, logged = solved
    assert solution.performance["memory_occupancy"] is None
    assert not any("memory_bytes" in slack for slack in solution.capacity_slack.values())
    assert solution.performance["overlap"]["per_resource_slack"]
    assert solution.slot_latency_breakdown["slots"]
    assert "memory occupancy" in logged
