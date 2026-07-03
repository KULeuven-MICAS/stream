"""Schema guard for the infeasibility diagnosis IR (stream.ir.infeasibility)."""

from __future__ import annotations

from stream.ir.infeasibility import (
    ImplicatedResourceIR,
    InfeasibilityReportIR,
    InfeasibleAllocationError,
    ResourceRefIR,
)


def _report() -> InfeasibilityReportIR:
    return InfeasibilityReportIR(
        status="INFEASIBLE",
        backend="GUROBI",
        solver="gurobi",
        group="Group_0_Frontend",
        iis_available=True,
        resources=[
            ImplicatedResourceIR(
                resource=ResourceRefIR(
                    kind="core", id="3", label="Core 3", detail={"memory_capacity_bits": "16777216"}
                ),
                constraint_kinds=["memory_capacity"],
                reason="on-chip memory capacity exceeded",
                constraints=["mem_cap_Core 3", "memload_x_Core_3_L-1__lb"],
            )
        ],
        unbound_constraints=["zStop_Choose_One_x"],
        summary="Infeasible mapping: on-chip memory capacity exceeded on Core 3",
    )


def test_report_json_roundtrip():
    report = _report()
    restored = InfeasibilityReportIR.model_validate_json(report.model_dump_json())
    assert restored.feasible is False
    assert restored.resources[0].resource.id == "3"
    assert restored.resources[0].resource.kind == "core"
    assert restored.resources[0].constraint_kinds == ["memory_capacity"]
    assert restored.iis_available is True


def test_error_carries_report():
    report = _report()
    err = InfeasibleAllocationError(report)
    assert isinstance(err, RuntimeError)  # callers catching RuntimeError still work
    assert err.report is report
    assert str(err) == report.summary


def test_resource_kind_is_open_for_new_hardware():
    """A future resource kind (memory bank, DMA engine, ...) needs no schema change."""
    ref = ResourceRefIR(kind="dma_engine", id="core3.dma0", label="DMA0 on Core 3")
    assert ref.kind == "dma_engine" and ref.detail == {}
