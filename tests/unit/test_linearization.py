"""Tests for the piecewise division linearization (D-09, D-10)."""

import pytest

from stream.opt.solver import (
    SolverBackend,
    SolverParams,
    SolverVarType,
    create_solver,
)


def _make_piecewise_model(numerator: float, denominators: list[float], force_selector: int):
    """Build a trivial model testing piecewise division.

    Creates binary selectors b_0..b_{n-1} (one-hot) with denominator values.
    Forces selector at index `force_selector` to 1.
    Result should equal numerator / denominators[force_selector].
    """
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "test_lin")
    m.set_param(SolverParams.VERBOSITY, 0)

    # Binary selectors (one-hot)
    b = [m.add_var(vtype=SolverVarType.BINARY, name=f"b_{i}") for i in range(len(denominators))]
    m.add_constr(m.quicksum(bi._raw for bi in b)._raw == 1, name="one_hot")

    # Force selector
    m.add_constr(b[force_selector]._raw == 1, name="force")

    # Piecewise division: result = sum(b_i * numerator / d_i)
    min_d = min(denominators)
    result = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=numerator / min_d, name="result")
    m.add_constr(
        result._raw == m.quicksum(b[i]._raw * (numerator / d) for i, d in enumerate(denominators))._raw,
        name="div_def",
    )
    m.set_objective(result._raw, sense="minimize")
    m.optimize()
    return m, result, b


@pytest.mark.parametrize(
    "force_idx, expected",
    [
        pytest.param(0, 50.0, id="d=2"),
        pytest.param(1, 25.0, id="d=4"),
        pytest.param(2, 12.5, id="d=8"),
    ],
)
def test_piecewise_division_matches_expected(force_idx, expected):
    m, result, _ = _make_piecewise_model(100.0, [2.0, 4.0, 8.0], force_idx)
    assert m.get_status() == "OPTIMAL", f"Expected OPTIMAL, got {m.get_status()}"
    assert abs(result.X - expected) < 1e-6, f"Expected {expected}, got {result.X}"


def test_binary_times_piecewise_division():
    """When binary gate y=0, result should be 0 regardless of selector."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "test_gated")
    m.set_param(SolverParams.VERBOSITY, 0)

    y = m.add_var(vtype=SolverVarType.BINARY, name="y")
    m.add_constr(y._raw == 0, name="force_y_zero")

    b = [m.add_var(vtype=SolverVarType.BINARY, name=f"b_{i}") for i in range(3)]
    m.add_constr(m.quicksum(bi._raw for bi in b)._raw == 1, name="one_hot")
    m.add_constr(b[0]._raw == 1, name="force_b0")

    numerator = 100.0
    denominators = [2.0, 4.0, 8.0]

    # Piecewise ratio
    ratio = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=50.0, name="ratio")
    m.add_constr(
        ratio._raw == m.quicksum(b[i]._raw * (numerator / d) for i, d in enumerate(denominators))._raw,
        name="ratio_def",
    )

    # Binary-scaled continuous: z = y * ratio
    z = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=50.0, name="z")
    m.add_constr(z._raw <= ratio._raw, name="prod_ub1")
    m.add_constr(z._raw <= 50.0 * y._raw, name="prod_ub2")
    m.add_constr(z._raw >= ratio._raw - 50.0 * (1 - y._raw), name="prod_lb1")
    m.add_constr(z._raw >= 0.0, name="prod_lb2")

    m.set_objective(z._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"
    assert abs(z.X) < 1e-6, f"Expected 0.0 when y=0, got {z.X}"


def test_binary_times_piecewise_division_active():
    """When binary gate y=1, result = ratio."""
    m = create_solver(SolverBackend.ORTOOLS_GSCIP, "test_gated_active")
    m.set_param(SolverParams.VERBOSITY, 0)

    y = m.add_var(vtype=SolverVarType.BINARY, name="y")
    m.add_constr(y._raw == 1, name="force_y_one")

    b = [m.add_var(vtype=SolverVarType.BINARY, name=f"b_{i}") for i in range(3)]
    m.add_constr(m.quicksum(bi._raw for bi in b)._raw == 1, name="one_hot")
    m.add_constr(b[1]._raw == 1, name="force_b1")

    numerator = 100.0
    denominators = [2.0, 4.0, 8.0]

    ratio = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=50.0, name="ratio")
    m.add_constr(
        ratio._raw == m.quicksum(b[i]._raw * (numerator / d) for i, d in enumerate(denominators))._raw,
        name="ratio_def",
    )

    z = m.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=50.0, name="z")
    m.add_constr(z._raw <= ratio._raw, name="prod_ub1")
    m.add_constr(z._raw <= 50.0 * y._raw, name="prod_ub2")
    m.add_constr(z._raw >= ratio._raw - 50.0 * (1 - y._raw), name="prod_lb1")
    m.add_constr(z._raw >= 0.0, name="prod_lb2")

    m.set_objective(z._raw, sense="minimize")
    m.optimize()
    assert m.get_status() == "OPTIMAL"
    assert abs(z.X - 25.0) < 1e-6, f"Expected 25.0 when y=1 and d=4, got {z.X}"
