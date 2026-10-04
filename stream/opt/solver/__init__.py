"""Solver abstraction package for TETRA constraint optimization.

Public API re-exported from stream.opt.solver.solver.
"""

from stream.opt.solver.solver import (
    GurobiBackend,
    LinExpr,
    ObjectiveLevel,
    ORToolsBackend,
    PipeliningModel,
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
    "PipeliningModel",
    "SolveStats",
    "SolverBackend",
    "SolverModel",
    "SolverParams",
    "SolverVar",
    "SolverVarType",
    "create_solver",
]
