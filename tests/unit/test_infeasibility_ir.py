"""Schema guard for the infeasibility diagnosis IR (stream.ir.infeasibility)."""

from __future__ import annotations

from types import SimpleNamespace

from stream.hardware.architecture.core import Core
from stream.ir.infeasibility import (
    ImplicatedResourceIR,
    InfeasibilityReportIR,
    InfeasibleAllocationError,
    ResourceRefIR,
)
from stream.opt.allocation.constraint_optimization.diagnosis import (
    ConstraintTag,
    ResourceKind,
    StructuralRule,
    infeasibility_report,
    structural_infeasibility,
)
from stream.opt.allocation.constraint_optimization.families.dma import DMA_CHANNELS
from stream.opt.allocation.constraint_optimization.families.memory import MEMORY_CAPACITY, OBJECT_FIFO_DEPTH
from stream.opt.allocation.constraint_optimization.formulation import ResourceLedger
from stream.opt.solver import SolverBackend, create_solver


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


def _core(core_id: int) -> Core:
    return Core(core_id=core_id, name=f"core_{core_id}", core_type="aie2.compute")


def _diagnose(ledger: ResourceLedger, iis: list[str]) -> InfeasibilityReportIR:
    """The diagnosis of a model whose IIS is the constraints ``iis``, from what ``ledger`` says they stand for."""
    stats = SimpleNamespace(backend="GUROBI", solver="gurobi")
    model = SimpleNamespace(
        solve_stats=lambda: stats, supports_iis=True, compute_iis=lambda: None, iis_constraints=lambda: iis
    )
    return infeasibility_report(model, ledger, "INFEASIBLE")  # type: ignore[arg-type]


def _bound(ledger: ResourceLedger, name: str, core: Core, kind: ResourceKind, bound: float) -> None:
    ledger.tags[name] = ConstraintTag(core, kind)
    ledger.bounds[(kind, core.id)] = bound


def _held(ledger: ResourceLedger, name: str, core: Core, kind: ResourceKind, tensor: str) -> None:
    ledger.tags[name] = ConstraintTag(core, kind, tensor)


def test_unmet_generalizes_beyond_memory():
    """The same builder quantifies a non-memory capacity family (object-FIFO depth) from the IIS
    witness -- proving the diagnosis is not memory-specific."""
    ledger, core = ResourceLedger(), _core(2)
    _bound(ledger, "fifo_bound", core, OBJECT_FIFO_DEPTH, 4.0)
    ledger.terms[(OBJECT_FIFO_DEPTH, 2)] = {"tA": 3, "tB": 3}
    _held(ledger, "a", core, OBJECT_FIFO_DEPTH, "tA")
    _held(ledger, "b", core, None, "tB")

    (resource,) = _diagnose(ledger, ["fifo_bound", "a", "b"]).resources
    unmet = resource.unmet
    assert unmet is not None
    assert resource.constraint_kinds == ["object_fifo_depth"]
    assert unmet.family == "object_fifo_depth"
    assert unmet.unit == "FIFO slots"
    assert unmet.bound_value == 4 and unmet.demand_value == 6 and unmet.gap == 2
    assert {t.label for t in unmet.terms} == {"tA", "tB"}
    assert "FIFO slots" in unmet.statement
    assert any("Core 2" in lever for lever in unmet.levers)


def test_unmet_forced_terms_avoid_partition_double_count():
    """Only the tensors the IIS constraints carry count, not every tensor whose name starts the same."""
    ledger, core = ResourceLedger(), _core(3)
    _bound(ledger, "mem_cap_Core 3", core, MEMORY_CAPACITY, 1_000_000.0)
    ledger.terms[(MEMORY_CAPACITY, 3)] = {"conv_out": 1_600_000, "conv_out_1": 1_600_000}
    _held(ledger, "memload_conv_out_1_Core_3_L-1__lb", core, MEMORY_CAPACITY, "conv_out_1")
    (resource,) = _diagnose(ledger, ["mem_cap_Core 3", "memload_conv_out_1_Core_3_L-1__lb"]).resources
    assert resource.unmet is not None
    assert {t.label for t in resource.unmet.terms} == {"conv_out_1"}
    assert resource.unmet.demand_value == 1_600_000


def test_unmet_memory_term_carries_tile_shape():
    """A memory term recorded as a {value, dims, dtype} record surfaces the per-dimension tile sizes and
    dtype, and the value still drives the demand -- so the designer sees which tile (and why) fills the
    core, not just the total."""
    ledger, core = ResourceLedger(), _core(3)
    _bound(ledger, "cap", core, MEMORY_CAPACITY, 1_000_000.0)
    ledger.terms[(MEMORY_CAPACITY, 3)] = {
        "conv_out": {"value": 1_600_000, "dims": [("z32", 64), ("z3", 112), ("z4", 112)], "dtype": "f32"}
    }
    _held(ledger, "load", core, MEMORY_CAPACITY, "conv_out")
    report = _diagnose(ledger, ["cap", "load"])
    assert report.nature == "capacity"
    (term,) = report.resources[0].unmet.terms  # type: ignore[union-attr]
    assert term.value == 1_600_000 and report.resources[0].unmet.demand_value == 1_600_000  # type: ignore[union-attr]
    assert term.dtype == "f32"
    assert [(d.label, d.size) for d in term.dims] == [("z32", 64), ("z3", 112), ("z4", 112)]


def test_a_constraint_is_diagnosed_by_its_tag_not_its_name():
    """A constraint whose name looks like a core's limit but carries no tag binds nothing, and a tagged one is
    attributed to its resource and limit whatever its name."""
    ledger, core = ResourceLedger(), _core(5)
    ledger.tags["anything"] = ConstraintTag(core, DMA_CHANNELS)
    report = _diagnose(ledger, ["mem_cap_Core 3", "dma_in_cap_Core 3", "anything"])
    assert report.unbound_constraints == ["mem_cap_Core 3", "dma_in_cap_Core 3"]
    (resource,) = report.resources
    assert (resource.resource.id, resource.constraint_kinds, resource.reason) == (
        "5",
        ["dma_channels"],
        "DMA channel limit exceeded",
    )


def test_a_structural_rule_in_the_iis_explains_the_conflict():
    ledger = ResourceLedger()
    rule = StructuralRule("Fused intermediate must stay resident", "It is re-read rather than spilled.")
    ledger.tags["held_long"] = ConstraintTag(rule=rule)
    report = _diagnose(ledger, ["held_long", "force_output_reuse_x"])
    assert report.nature == "structural"
    assert [(c.title, c.constraints) for c in report.conflicts] == [(rule.title, ["held_long"])]


def test_a_structural_failure_before_any_solve_names_the_backend():
    model = create_solver(SolverBackend.ORTOOLS_GSCIP)
    assert structural_infeasibility("no core", model).backend == "ORTOOLS_GSCIP"
    assert structural_infeasibility("no core").backend == "n/a"
