"""The stats of a lexicographic solve report the level asked for, not the last one optimized: minimize x + y (the
latency level) over x, y in [0, 10] with x + y >= 5, then x; the first level settles at 5, the second at 0."""

from __future__ import annotations

import pytest

from stream.opt.solver import ObjectiveLevel, SolverBackend, SolverParams, SolverVarType, create_solver


def _solved(backend: SolverBackend):
    model = create_solver(backend, "levels")
    model.set_param(SolverParams.VERBOSITY, 0)
    x = model.add_var(vtype=SolverVarType.INTEGER, ub=10, name="x")
    y = model.add_var(vtype=SolverVarType.INTEGER, ub=10, name="y")
    model.add_constr(x + y >= 5)
    model.set_lexicographic_objectives(
        [ObjectiveLevel(expr=x + y, priority=2, name="latency"), ObjectiveLevel(expr=x, priority=1, name="second")]
    )
    model.optimize()
    return model


def test_or_tools_reports_the_level_asked_for():
    model = _solved(SolverBackend.ORTOOLS_GSCIP)
    latency, last = model.solve_stats(level="latency"), model.solve_stats()
    assert (latency.objective, latency.mip_gap, last.objective) == (5, 0, 0)
    assert latency.solve_time_s == last.solve_time_s > 0


def test_gurobi_reports_the_level_asked_for_and_no_gap_for_it():
    """Gurobi's multi-objective solve defines no gap per level."""
    pytest.importorskip("gurobipy")
    stats = _solved(SolverBackend.GUROBI).solve_stats(level="latency")
    assert (stats.objective, stats.mip_gap) == (5, None)
