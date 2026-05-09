"""Unit tests for ORToolsBackend (Phase 2).

Covers ORT-01 (all linear MILP operations) and ORT-03 (infeasibility + MPS export).
"""

import os
import tempfile
from collections import defaultdict

import pytest
from ortools.math_opt.python import mathopt

from stream.opt.solver import (
    LinExpr,
    ORToolsBackend,
    SolverBackend,
    SolverModel,
    SolverParams,
    SolverVar,
    SolverVarType,
    create_solver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ortools_trivial_model() -> tuple[SolverModel, SolverVar]:
    """Create a trivial LP: minimize x subject to 0 <= x <= 10, using ORToolsBackend."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "trivial")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    m.set_objective(x._raw, sense="minimize")
    return m, x


def _ortools_infeasible_model() -> tuple[SolverModel, SolverVar]:
    """Create an infeasible LP: x >= 5 and x <= 3, minimize x."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "infeasible")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    m.add_constr(x._raw >= 5.0, name="lb_infeasible")
    m.add_constr(x._raw <= 3.0, name="ub_infeasible")
    m.set_objective(x._raw, sense="minimize")
    return m, x


# ---------------------------------------------------------------------------
# ORT-01: Variable creation
# ---------------------------------------------------------------------------


def test_ortools_add_binary_var():
    """add_var(vtype=BINARY) returns SolverVar; ._raw is mathopt.Variable."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "binary_test")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.BINARY, name="x")
    assert isinstance(x, SolverVar), f"Expected SolverVar, got {type(x)}"
    assert isinstance(x._raw, mathopt.Variable), f"Expected mathopt.Variable, got {type(x._raw)}"


def test_ortools_add_continuous_var():
    """add_var(vtype=CONTINUOUS, lb=0, ub=10) returns SolverVar."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "continuous_test")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    assert isinstance(x, SolverVar), f"Expected SolverVar, got {type(x)}"


def test_ortools_add_integer_var():
    """add_var(vtype=INTEGER, lb=0, ub=5) returns SolverVar."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "integer_test")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.INTEGER, lb=0, ub=5, name="x")
    assert isinstance(x, SolverVar), f"Expected SolverVar, got {type(x)}"


# ---------------------------------------------------------------------------
# ORT-01: Constraints and solve
# ---------------------------------------------------------------------------


def test_ortools_trivial_lp():
    """Trivial LP: minimize x s.t. 0 <= x <= 10 => OPTIMAL, x.X == 0.0."""
    m, x = _ortools_trivial_model()
    m.optimize()
    status = m.get_status()
    assert status == "OPTIMAL", f"Expected OPTIMAL, got {status!r}"
    assert abs(x.X - 0.0) < 1e-6, f"Expected x.X == 0.0, got {x.X}"


def test_ortools_lp_with_constraint():
    """LP with lower-bound constraint: x >= 3 => x.X == 3.0."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "constrained")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    m.add_constr(x._raw >= 3.0, name="lb_constr")
    m.set_objective(x._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL", f"Expected OPTIMAL, got {m.get_status()}"
    assert abs(x.X - 3.0) < 1e-6, f"Expected x.X == 3.0, got {x.X}"


def test_ortools_linear_constraints():
    """<=, >=, == constraints all work without raising."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "constr_types")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    y = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="y")
    # These must not raise
    m.add_constr(x._raw <= 5.0, name="le_constr")
    m.add_constr(y._raw >= 2.0, name="ge_constr")
    m.add_constr(x._raw + y._raw == 7.0, name="eq_constr")
    m.set_objective(x._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"


def test_ortools_binary_milp():
    """Maximize x+y (binary) s.t. x+y<=1 => obj == 1.0."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "binary_milp")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.BINARY, name="x")
    y = m.add_var(vtype=SolverVarType.BINARY, name="y")
    m.add_constr(x._raw + y._raw <= 1.0, name="c1")
    m.set_objective(x._raw + y._raw, sense="maximize")
    m.optimize()
    assert m.get_status() == "OPTIMAL", f"Expected OPTIMAL, got {m.get_status()}"
    assert abs(x.X + y.X - 1.0) < 1e-6, f"Expected x+y == 1.0, got {x.X + y.X}"


# ---------------------------------------------------------------------------
# ORT-01: Expression building
# ---------------------------------------------------------------------------


def test_ortools_quicksum():
    """quicksum([v1._raw, v2._raw]) returns LinExpr, usable in constraint."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "quicksum_test")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=5.0, name="x")
    y = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=5.0, name="y")
    total = m.quicksum([x._raw, y._raw])
    assert isinstance(total, LinExpr), f"Expected LinExpr, got {type(total)}"
    # usable in constraint
    m.add_constr(total <= 8.0, name="sum_constr")
    m.set_objective(x._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"


def test_ortools_quicksum_mixed():
    """quicksum with LinExpr + int items does not raise (regression analog)."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "quicksum_mixed")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.BINARY, name="x")
    y = m.add_var(vtype=SolverVarType.BINARY, name="y")
    # First-level quicksum
    inner = m.quicksum([x._raw, y._raw])
    # Second-level with mixed LinExpr and int
    outer = m.quicksum([inner, 1])
    assert outer is not None, "quicksum result should not be None"
    # Must be usable in a constraint (use outer._raw to unwrap for the native operator)
    z = m.add_var(vtype=SolverVarType.INTEGER, lb=0, ub=10, name="z")
    m.add_constr(z._raw >= outer._raw, name="mixed_constr")
    m.set_objective(z._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"


def test_ortools_linexpr_accumulation():
    """lin_expr() += var._raw multiple times, then use in constraint."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "linexpr_accum")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=5.0, name="x")
    y = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=5.0, name="y")
    expr = m.lin_expr()
    expr += x._raw
    expr += y._raw
    m.add_constr(expr <= 7.0, name="accum_constr")
    m.set_objective(x._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"


def test_ortools_linexpr_defaultdict():
    """defaultdict(model.lin_expr) pattern works."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "defaultdict_test")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=5.0, name="x")
    y = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=5.0, name="y")
    d: defaultdict = defaultdict(m.lin_expr)
    d["k"] += x._raw
    d["k"] += y._raw
    # Must be usable in a constraint
    m.add_constr(d["k"] <= 8.0, name="defaultdict_constr")
    m.set_objective(x._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"


# ---------------------------------------------------------------------------
# ORT-01: Solution extraction
# ---------------------------------------------------------------------------


def test_ortools_var_x():
    """After solve, var.X returns float solution value."""
    m, x = _ortools_trivial_model()
    m.optimize()
    val = x.X
    assert isinstance(val, float), f"SolverVar.X should return float, got {type(val)}"


def test_ortools_var_x_before_solve():
    """Accessing .X before optimize() raises ValueError."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "pre_solve")
    m.set_param(SolverParams.VERBOSITY, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    with pytest.raises(ValueError):
        _ = x.X


# ---------------------------------------------------------------------------
# ORT-01: Parameters
# ---------------------------------------------------------------------------


def test_ortools_set_param():
    """set_param(VERBOSITY, 0), set_param(TIME_LIMIT, 60), set_param(THREADS, 1) do not raise."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "param_test")
    m.set_param(SolverParams.VERBOSITY, 0)
    m.set_param(SolverParams.TIME_LIMIT, 60)
    m.set_param(SolverParams.THREADS, 1)


# ---------------------------------------------------------------------------
# ORT-01: Factory
# ---------------------------------------------------------------------------


def test_factory_ortools():
    """create_solver(SolverBackend.ORTOOLS_GSCIP) returns ORToolsBackend."""
    solver = create_solver(SolverBackend.ORTOOLS_GSCIP, "factory_test")
    assert isinstance(solver, ORToolsBackend), f"Expected ORToolsBackend, got {type(solver)}"
    assert isinstance(solver, SolverModel), "ORToolsBackend should be a SolverModel"


def test_factory_ortools_highs():
    """create_solver(ORTOOLS_HIGHS) returns ORToolsBackend with HiGHS solver."""
    solver = create_solver(SolverBackend.ORTOOLS_HIGHS, "highs_test")
    assert isinstance(solver, ORToolsBackend), f"Expected ORToolsBackend, got {type(solver)}"


def test_factory_ortools_gurobi():
    """create_solver(ORTOOLS_GUROBI) returns ORToolsBackend with Gurobi solver."""
    solver = create_solver(SolverBackend.ORTOOLS_GUROBI, "gurobi_ort_test")
    assert isinstance(solver, ORToolsBackend), f"ORTOOLS_GUROBI should route to ORToolsBackend, got {type(solver)}"


# ---------------------------------------------------------------------------
# ORT-01: Status
# ---------------------------------------------------------------------------


def test_ortools_get_status_optimal():
    """After successful solve, get_status() == 'OPTIMAL'."""
    m, _ = _ortools_trivial_model()
    m.optimize()
    assert m.get_status() == "OPTIMAL", f"Expected OPTIMAL, got {m.get_status()!r}"


def test_ortools_get_sol_count():
    """After successful solve, get_sol_count() == 1."""
    m, _ = _ortools_trivial_model()
    m.optimize()
    assert m.get_sol_count() == 1, f"Expected sol_count == 1, got {m.get_sol_count()}"


# ---------------------------------------------------------------------------
# ORT-03: Infeasibility
# ---------------------------------------------------------------------------


def test_ortools_infeasible_detection():
    """Infeasible model => get_status() == 'INFEASIBLE'."""
    m, _ = _ortools_infeasible_model()
    m.optimize()
    assert m.get_status() == "INFEASIBLE", f"Expected INFEASIBLE, got {m.get_status()!r}"


def test_ortools_mps_export():
    """write(path) creates a file containing valid MPS content."""
    m, _ = _ortools_trivial_model()
    with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        m.write(tmp_path)
        assert os.path.exists(tmp_path), f"MPS file not created at {tmp_path}"
        with open(tmp_path) as f:
            content = f.read()
        assert len(content) > 0, "MPS file is empty"
        # MPS format starts with NAME section
        assert "NAME" in content, f"MPS content missing NAME section. Got: {content[:200]!r}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_ortools_compute_iis_no_raise():
    """compute_iis() on infeasible model does NOT raise (logs warning instead)."""
    m, _ = _ortools_infeasible_model()
    m.optimize()
    # Must NOT raise any exception
    m.compute_iis()


def test_ortools_infeasible_var_x_raises():
    """Accessing .X on infeasible result raises ValueError."""
    m, x = _ortools_infeasible_model()
    m.optimize()
    with pytest.raises(ValueError):
        _ = x.X


# ---------------------------------------------------------------------------
# Cross-backend: Objective equivalence
# ---------------------------------------------------------------------------


def test_cross_backend_objective():
    """Same trivial MILP on GSCIP and HiGHS produces the same objective value."""

    def build_and_solve(backend_enum: SolverBackend) -> float:
        m = create_solver(backend_enum, "cross_backend")
        m.set_param(SolverParams.VERBOSITY, 0)
        x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=100.0, name="x")
        y = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=100.0, name="y")
        m.add_constr(x._raw + y._raw >= 5.0, name="sum_lb")
        m.set_objective(2.0 * x._raw + 3.0 * y._raw, sense="minimize")
        m.optimize()
        assert m.get_status() == "OPTIMAL", f"Expected OPTIMAL on {backend_enum}, got {m.get_status()}"
        return x.X * 2.0 + y.X * 3.0

    gscip_obj = build_and_solve(SolverBackend.ORTOOLS_GSCIP)
    highs_obj = build_and_solve(SolverBackend.ORTOOLS_HIGHS)
    assert abs(gscip_obj - highs_obj) < 1e-4, f"Cross-backend mismatch: GSCIP={gscip_obj}, HiGHS={highs_obj}"
