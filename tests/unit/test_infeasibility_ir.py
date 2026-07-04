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


def _bare_allocator():
    """A TransferAndTensorAllocator with only the quantitative-diagnosis state populated -- enough to
    exercise the family-agnostic _build_unmet without constructing a full MILP."""
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    alloc = object.__new__(TransferAndTensorAllocator)
    alloc._resource_bounds = {}
    alloc._resource_terms = {}
    return alloc


def test_unmet_generalizes_beyond_memory():
    """The same builder quantifies a non-memory capacity family (object-FIFO depth) from the IIS
    witness -- proving the diagnosis is not memory-specific."""
    alloc = _bare_allocator()
    alloc._resource_bounds[("object_fifo_depth", 2)] = 4.0
    alloc._resource_terms[("object_fifo_depth", 2)] = {"tA": 3, "tB": 3}
    iis = ["aie2_obj_fifo_depth_Core_2", "objfifo_tA_Core_2_L-1__lb", "objfifo_tB_Core_2_L0__lb"]

    unmet = alloc._build_unmet("object_fifo_depth", 2, iis)
    assert unmet is not None
    assert unmet.family == "object_fifo_depth"
    assert unmet.unit == "FIFO slots"
    assert unmet.bound_value == 4 and unmet.demand_value == 6 and unmet.gap == 2
    assert {t.label for t in unmet.terms} == {"tA", "tB"}
    assert "FIFO slots" in unmet.statement
    assert any("Core 2" in lever for lever in unmet.levers)


def test_unmet_forced_terms_avoid_partition_double_count():
    """A base tensor name is a prefix of its partitions; only the exact IIS subjects count."""
    alloc = _bare_allocator()
    alloc._resource_bounds[("memory_capacity", 3)] = 2_000_000.0
    alloc._resource_terms[("memory_capacity", 3)] = {"conv_out": 1_600_000, "conv_out_1": 1_600_000}
    # Only the "_1" partition is in the IIS -> the base must NOT be double-counted.
    iis = ["mem_cap_Core 3", "memload_conv_out_1_Core_3_L-1__lb"]
    unmet = alloc._build_unmet("memory_capacity", 3, iis)
    assert unmet is not None
    assert {t.label for t in unmet.terms} == {"conv_out_1"}
    assert unmet.demand_value == 1_600_000
