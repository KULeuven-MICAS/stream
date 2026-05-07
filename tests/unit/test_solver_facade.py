"""Unit tests for the solver abstraction facade.

Covers ABS-01 through ABS-05 from the Phase 1 requirements:
  - ABS-01: SolverModel ABC cannot be instantiated directly
  - ABS-02: GurobiBackend implements SolverModel correctly
  - ABS-03: SolverVar exposes .X and delegates arithmetic operators
  - ABS-04: LinExpr supports += and + operations and defaultdict patterns
  - ABS-05: Factory dispatches to the correct backend or raises

All tests use the public API from stream.opt.solver (not the private submodule).
"""

from collections import defaultdict

import pytest

from stream.opt.solver import (
    GurobiBackend,
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


def _trivial_model() -> tuple[SolverModel, SolverVar]:
    """Create a trivial LP: minimize x subject to 0 <= x <= 10."""
    m = create_solver(SolverBackend.GUROBI, "trivial")
    m.set_param(SolverParams.VERBOSITY, 0)
    m.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    m.set_objective(x._raw, sense="minimize")
    return m, x


# ---------------------------------------------------------------------------
# ABS-01: SolverModel ABC contract
# ---------------------------------------------------------------------------


def test_abc_not_instantiable():
    """SolverModel is abstract and cannot be directly instantiated."""
    with pytest.raises(TypeError):
        SolverModel()  # type: ignore[abstract]


def test_abc_interface():
    """SolverModel defines all required abstract methods."""
    required_abstract = {
        "add_var",
        "add_constr",
        "set_objective",
        "optimize",
        "set_param",
        "get_status",
        "get_sol_count",
        "compute_iis",
        "write",
        "quicksum",
        "lin_expr",
    }
    abstract_methods = getattr(SolverModel, "__abstractmethods__", frozenset())
    for method in required_abstract:
        assert method in abstract_methods, (
            f"SolverModel.{method} should be abstract but is not in __abstractmethods__: {abstract_methods}"
        )


# ---------------------------------------------------------------------------
# ABS-02: GurobiBackend implements SolverModel
# ---------------------------------------------------------------------------


def test_gurobi_add_var():
    """GurobiBackend.add_var returns a SolverVar instance."""
    model = GurobiBackend("test_add_var")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    var = model.add_var(vtype=SolverVarType.BINARY, lb=0, ub=1, name="x")
    assert isinstance(var, SolverVar), f"Expected SolverVar, got {type(var)}"


def test_gurobi_trivial_lp():
    """GurobiBackend solves a trivial LP and reports OPTIMAL status."""
    m, x = _trivial_model()
    m.optimize()
    status = m.get_status()
    assert status == "OPTIMAL", f"Expected OPTIMAL status, got {status!r}"
    assert abs(x.X - 0.0) < 1e-6, f"Expected x=0.0 after minimization, got {x.X}"


def test_gurobi_trivial_lp_with_constraint():
    """GurobiBackend solves LP with a lower-bound constraint correctly."""
    m = create_solver(SolverBackend.GUROBI, "constrained")
    m.set_param(SolverParams.VERBOSITY, 0)
    m.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    x = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    # x >= 3
    m.add_constr(x._raw >= 3.0, name="lb_constr")
    m.set_objective(x._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL", f"Expected OPTIMAL, got {m.get_status()}"
    assert abs(x.X - 3.0) < 1e-6, f"Expected x=3.0, got {x.X}"


# ---------------------------------------------------------------------------
# ABS-03: SolverVar interface
# ---------------------------------------------------------------------------


def test_solver_var_x():
    """SolverVar.X returns the correct solution value after solve."""
    m, x = _trivial_model()
    m.optimize()
    val = x.X
    assert isinstance(val, float), f"SolverVar.X should return float, got {type(val)}"
    assert abs(val - 0.0) < 1e-6, f"Expected x.X == 0.0 after minimization, got {val}"


def test_solver_var_arithmetic():
    """SolverVar._raw returns underlying gp.Var; arithmetic operators work."""
    import gurobipy as gp

    model = GurobiBackend("arith_test")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    var = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="v")

    # ._raw should return a gp.Var
    assert isinstance(var._raw, gp.Var), f"Expected gp.Var, got {type(var._raw)}"

    # var + 1 should produce a LinExpr, not raise
    expr_add = var + 1
    assert expr_add is not None, "var + 1 should not return None"

    # var <= 5 should produce a valid Gurobi constraint expression, not a bool
    constr_expr = var <= 5
    assert not isinstance(constr_expr, bool), (
        f"var <= 5 should produce a constraint expression, not a Python bool. "
        f"Got type {type(constr_expr)}"
    )


def test_solver_var_arithmetic_operators():
    """SolverVar supports all arithmetic operators without raising."""
    model = GurobiBackend("ops_test")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    x = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="x")
    y = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="y")

    # These should not raise
    _ = x + y
    _ = x + 1
    _ = 1 + x
    _ = x - y
    _ = x - 1
    _ = 1 - x
    _ = x * 2
    _ = 2 * x
    _ = -x
    _ = x <= y
    _ = x >= y
    _ = x == y


# ---------------------------------------------------------------------------
# ABS-04: LinExpr interface
# ---------------------------------------------------------------------------


def test_linexpr_ops():
    """LinExpr supports += and + operations without raising."""
    model = GurobiBackend("linexpr_test")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    var = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="v")

    expr = model.lin_expr()
    expr += var._raw
    assert expr is not None, "LinExpr after += should not be None"

    expr2 = expr + var._raw
    assert expr2 is not None, "LinExpr + var._raw should not be None"


def test_linexpr_defaultdict():
    """LinExpr works as a defaultdict(model.lin_expr) value factory."""
    model = GurobiBackend("defaultdict_test")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    var = model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=10.0, name="v")

    d: defaultdict = defaultdict(model.lin_expr)
    # First access auto-creates the LinExpr; += accumulates the var
    d["key"] += var._raw
    assert d["key"] is not None, "defaultdict value after += should not be None"

    # Multiple accumulations should not raise
    d["key"] += var._raw
    d["key2"] += var._raw


# ---------------------------------------------------------------------------
# ABS-05: Factory function
# ---------------------------------------------------------------------------


def test_factory_gurobi():
    """create_solver(GUROBI) returns a GurobiBackend instance."""
    solver = create_solver(SolverBackend.GUROBI, "test")
    assert isinstance(solver, GurobiBackend), (
        f"Expected GurobiBackend, got {type(solver)}"
    )


def test_factory_scip_raises():
    """create_solver(SCIP) raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="SCIP"):
        create_solver(SolverBackend.SCIP)


def test_factory_ortools_raises():
    """create_solver(ORTOOLS_GUROBI) raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        create_solver(SolverBackend.ORTOOLS_GUROBI)


def test_factory_unknown_raises():
    """create_solver with an unknown value raises ValueError (or TypeError)."""
    with pytest.raises((ValueError, TypeError)):
        create_solver("nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


def test_set_param_verbosity():
    """set_param(VERBOSITY, 0) does not raise."""
    model = create_solver(SolverBackend.GUROBI, "param_test")
    model.set_param(SolverParams.VERBOSITY, 0)  # Should not raise


def test_get_status_before_solve():
    """get_status() returns a string even before solving."""
    model = create_solver(SolverBackend.GUROBI, "status_test")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
    status = model.get_status()
    assert isinstance(status, str), (
        f"get_status() should return a string, got {type(status)} ({status!r})"
    )


def test_quicksum_with_mixed_types():
    """quicksum handles mixed SolverVar, LinExpr, and int items without error.

    Regression test for the bug where quicksum of _GurobiLinExpr items
    produced an invalid nested expression that caused addConstr to fail
    with 'Invalid argument to Model.addConstr'.
    """
    model = create_solver(SolverBackend.GUROBI, "quicksum_test")
    model.set_param(SolverParams.VERBOSITY, 0)
    model.set_param(SolverParams.LOG_TO_CONSOLE, 0)

    x1 = model.add_var(vtype=SolverVarType.BINARY, name="x1")
    x2 = model.add_var(vtype=SolverVarType.BINARY, name="x2")

    # First-level quicksum produces a LinExpr (SolverVar._raw items)
    inner = model.quicksum([x1._raw, x2._raw])

    # Second-level quicksum with mixed LinExpr and int — triggers the bug if not fixed
    outer = model.quicksum([inner, 1])

    # The outer result's ._raw must be a raw gp.LinExpr, not a wrapped _GurobiLinExpr
    import gurobipy as gp

    assert isinstance(outer._raw, gp.LinExpr), (
        f"quicksum result._raw must be a gp.LinExpr, got {type(outer._raw)}"
    )

    # Should be usable in a constraint without 'Invalid argument' error
    v = model.add_var(vtype=SolverVarType.INTEGER, lb=0, name="v")
    model.add_constr(v == outer, name="mixed_quicksum_constr")  # Must not raise
