"""Solver abstraction package for TETRA constraint optimization.

Public API re-exported from stream.opt.solver.solver.
"""

from stream.opt.solver.solver import (
    ConstraintSelection,
    GurobiBackend,
    LinExpr,
    ObjectiveLevel,
    ORToolsBackend,
    SolverBackend,
    SolverModel,
    SolverParams,
    SolverVar,
    SolverVarType,
    SolveStats,
    create_solver,
)

__all__ = [
    "ConstraintSelection",
    "GurobiBackend",
    "LinExpr",
    "ObjectiveLevel",
    "ORToolsBackend",
    "SolveStats",
    "SolverBackend",
    "SolverModel",
    "SolverParams",
    "SolverVar",
    "SolverVarType",
    "create_solver",
]
