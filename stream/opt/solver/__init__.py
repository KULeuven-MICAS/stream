"""Solver abstraction package for the allocation model.

Public API re-exported from stream.opt.solver.solver.
"""

from stream.opt.solver.solver import (
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
