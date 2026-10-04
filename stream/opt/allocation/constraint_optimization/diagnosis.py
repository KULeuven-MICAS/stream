"""Why an allocation model has no solution, as an inspectable report rather than a bare failure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stream.ir.infeasibility import InfeasibilityReportIR

if TYPE_CHECKING:
    from stream.opt.solver import SolverModel


def structural_infeasibility(reason: str, model: SolverModel | None = None) -> InfeasibilityReportIR:
    """A minimal infeasibility report for a structural problem in the mapping itself (a node with no
    valid core), raised during model construction -- so an unbuildable model fails with an
    inspectable diagnosis rather than a bare exception."""
    backend = solver = "n/a"
    if model is not None:
        try:
            stats = model.solve_stats()
            backend, solver = stats.backend, stats.solver
        except Exception:  # noqa: BLE001 -- solve stats may be unavailable before the first solve
            pass
    return InfeasibilityReportIR(
        status="INFEASIBLE",
        backend=backend,
        solver=solver,
        group=None,
        iis_available=False,
        nature="structural",
        resources=[],
        unbound_constraints=[reason],
        summary=(
            f"Infeasible mapping: {reason}. The auto-generated mapping could not place every tensor on "
            "this hardware -- it likely needs a hand-written mapping."
        ),
    )
