"""Solver abstraction layer for TETRA constraint optimization.

Provides a backend-agnostic API over MILP solvers. Currently supports
GurobiBackend (wrapping gurobipy 13.0.0) and ORToolsBackend (wrapping
OR-Tools MathOpt API, Phase 2).

Design decisions:
- D-01: Core ops only — ABC wraps addVar, addConstr, setObjective, optimize, and solution extraction.
- D-05: quicksum as a method on SolverModel.
- D-06: Lives in stream/opt/solver/ (new package under opt/).
- D-07: Single module layout: solver.py contains ABC, GurobiBackend, SolverVar, LinExpr, enums, factory.
"""

from __future__ import annotations

import datetime
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:  # gurobipy is optional — only the Gurobi backend needs it.
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SolverVarType(Enum):
    """Variable type for solver variables."""

    BINARY = auto()
    INTEGER = auto()
    CONTINUOUS = auto()


class SolverParams(Enum):
    """Solver configuration parameters.

    Each backend maps these to its native parameter API.
    Not all backends support all parameters — unsupported params
    raise NotImplementedError.
    """

    VERBOSITY = auto()
    TIME_LIMIT = auto()
    THREADS = auto()
    POOL_GAP = auto()
    LOG_TO_CONSOLE = auto()


class SolverBackend(Enum):
    """Available solver backends.

    GUROBI: Native gurobipy (requires license).
    ORTOOLS_GSCIP: MathOpt with GSCIP solver (open-source, bundled).
    ORTOOLS_HIGHS: MathOpt with HiGHS solver (open-source, bundled).
    ORTOOLS_GUROBI: MathOpt using Gurobi as backend (requires license).
    """

    GUROBI = "GUROBI"
    ORTOOLS_GSCIP = "ORTOOLS_GSCIP"
    ORTOOLS_HIGHS = "ORTOOLS_HIGHS"
    ORTOOLS_GUROBI = "ORTOOLS_GUROBI"


@dataclass(frozen=True)
class SolveStats:
    """Structured solve statistics returned by SolverModel.solve_stats().

    All fields are populated after optimize() has been called.
    Fields not available for a given backend are set to None.
    """

    backend: str
    """Backend name, e.g. 'GUROBI' or 'ORTOOLS'."""
    solver: str
    """Underlying solver name, e.g. 'gurobi', 'gscip', 'highs'."""
    status: str
    """Solve status string, e.g. 'OPTIMAL', 'INFEASIBLE', 'TIME_LIMIT'."""
    objective: float | None
    """Objective value of best solution found, or None if no solution."""
    solve_time_s: float
    """Wall-clock solve time in seconds."""
    mip_gap: float | None
    """Relative MIP gap, or None if not available for this backend."""
    node_count: int | None
    """Number of B&B nodes explored, or None if not available for this backend."""
    iteration_count: int | None
    """Number of simplex iterations, or None if not available for this backend."""


@dataclass(frozen=True)
class ConstraintSelection:
    """Toggle hardware resource constraint groups in TransferAndTensorAllocator.

    All groups default to True (fully constrained). Set a field to False
    to skip that constraint group entirely -- no variables are created,
    no constraints are added, and (for dma_channels) objective terms are omitted.
    """

    memory_capacity: bool = True
    object_fifo_depth: bool = True
    buffer_descriptors: bool = True
    dma_channels: bool = True

    def __post_init__(self) -> None:
        if not self.memory_capacity and self.object_fifo_depth:
            _logger.warning(
                "ConstraintSelection: memory_capacity=False with object_fifo_depth=True "
                "is nonsensical -- object-FIFO depth constraints assume memory capacity "
                "is enforced. Continuing with this configuration."
            )


@dataclass(frozen=True)
class ObjectiveLevel:
    """A single level in a lexicographic objective hierarchy.

    Higher ``priority`` objectives are optimized first.  Lower-priority
    objectives are optimized subject to the constraint that all
    higher-priority objectives do not degrade beyond the specified
    tolerances.

    Attributes:
        expr: Backend expression for this objective (raw or wrapped).
        priority: Priority of this level (higher = optimized first).
        name: Human-readable name for logging and debugging.
        abs_tol: Absolute degradation tolerance when locking this level
            for lower-priority objectives.
        rel_tol: Relative degradation tolerance (fraction of optimal value)
            when locking this level for lower-priority objectives.
    """

    expr: Any
    priority: int
    name: str = ""
    abs_tol: float = 0.0
    rel_tol: float = 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _unwrap(other: Any) -> Any:
    """Unwrap SolverVar or LinExpr to underlying backend object.

    Passes through int, float, gp.Var, gp.LinExpr, and other raw types.
    """
    if isinstance(other, (_GurobiVar, _GurobiLinExpr)):
        return other._raw
    return other


# ---------------------------------------------------------------------------
# SolverVar ABC
# ---------------------------------------------------------------------------


class SolverVar(ABC):
    """Backend-agnostic wrapper around a solver decision variable.

    Exposes .X for solution extraction and ._raw for building backend-
    specific constraint expressions.

    Per D-01 and ABS-03: all CO code that builds constraint expressions
    should use var._raw (for GurobiBackend this is a gp.Var). Arithmetic
    operator delegation is also provided so SolverVar objects can be used
    directly in expressions — they delegate to ._raw.
    """

    @property
    @abstractmethod
    def X(self) -> float:
        """Solution value after solve."""

    @property
    @abstractmethod
    def _raw(self) -> Any:
        """Underlying backend variable for expression building."""

    # Arithmetic operator delegation — all abstract, implemented by subclasses
    @abstractmethod
    def __le__(self, other: Any) -> Any: ...

    @abstractmethod
    def __ge__(self, other: Any) -> Any: ...

    @abstractmethod
    def __eq__(self, other: Any) -> Any: ...

    @abstractmethod
    def __ne__(self, other: Any) -> Any: ...

    @abstractmethod
    def __add__(self, other: Any) -> Any: ...

    @abstractmethod
    def __radd__(self, other: Any) -> Any: ...

    @abstractmethod
    def __sub__(self, other: Any) -> Any: ...

    @abstractmethod
    def __rsub__(self, other: Any) -> Any: ...

    @abstractmethod
    def __mul__(self, other: Any) -> Any: ...

    @abstractmethod
    def __rmul__(self, other: Any) -> Any: ...

    @abstractmethod
    def __truediv__(self, other: Any) -> Any: ...

    @abstractmethod
    def __rtruediv__(self, other: Any) -> Any: ...

    @abstractmethod
    def __neg__(self) -> Any: ...

    @abstractmethod
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# LinExpr ABC
# ---------------------------------------------------------------------------


class LinExpr(ABC):
    """Backend-agnostic linear expression.

    Supports += and + accumulation patterns, including use as a
    defaultdict(model.lin_expr) value factory.

    Per D-01 and ABS-04: GurobiBackend wraps gp.LinExpr.
    """

    @property
    @abstractmethod
    def _raw(self) -> Any:
        """Underlying backend expression object."""

    @abstractmethod
    def __iadd__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __add__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __radd__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __sub__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __rsub__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __mul__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __rmul__(self, other: Any) -> LinExpr: ...

    @abstractmethod
    def __neg__(self) -> LinExpr: ...

    @abstractmethod
    def __le__(self, other: Any) -> Any: ...

    @abstractmethod
    def __ge__(self, other: Any) -> Any: ...

    @abstractmethod
    def __eq__(self, other: Any) -> Any: ...


# ---------------------------------------------------------------------------
# SolverModel ABC
# ---------------------------------------------------------------------------


class SolverModel(ABC):
    """Abstract base class for MILP solver backends.

    Per D-01 and ABS-01: defines the interface for variable creation,
    constraint addition, objective setting, and solving. Does NOT wrap
    tupledict, addVars, addConstrs, multidict, max_, min_.

    Class constants:
        INFINITY: float  — use as upper bound for unbounded variables
        MINIMIZE: str    — sense string for minimize objective
        MAXIMIZE: str    — sense string for maximize objective
    """

    INFINITY: float
    MINIMIZE: str = "minimize"
    MAXIMIZE: str = "maximize"

    @abstractmethod
    def add_var(
        self,
        *,
        vtype: SolverVarType,
        lb: float = 0.0,
        ub: float | None = None,
        name: str = "",
    ) -> SolverVar:
        """Create a new decision variable.

        Args:
            vtype: Variable type (BINARY, INTEGER, or CONTINUOUS).
            lb: Lower bound (default 0.0).
            ub: Upper bound (None means INFINITY).
            name: Variable name for debugging.

        Returns:
            A SolverVar wrapping the backend variable.
        """

    @abstractmethod
    def add_constr(self, expr: Any, *, name: str = "") -> None:
        """Add a constraint to the model.

        Args:
            expr: Backend constraint expression (e.g. TempConstr for Gurobi).
            name: Constraint name for debugging.
        """

    @abstractmethod
    def set_objective(self, expr: Any, *, sense: str = "minimize") -> None:
        """Set the optimization objective.

        Args:
            expr: Backend expression for the objective.
            sense: "minimize" or "maximize".
        """

    @abstractmethod
    def optimize(self, callback: Any = None) -> None:
        """Run the optimizer.

        Args:
            callback: Optional backend-specific callback (D-04).
                      GurobiBackend passes this to model.optimize(callback).
                      Non-Gurobi backends may ignore it.
        """

    @abstractmethod
    def set_param(self, param: SolverParams, value: Any) -> None:
        """Set a solver parameter.

        Raises:
            NotImplementedError: If the parameter is not supported by this backend.
        """

    @abstractmethod
    def get_status(self) -> str:
        """Return the solve status as a string.

        Returns:
            One of: "OPTIMAL", "INFEASIBLE", "TIME_LIMIT", "UNBOUNDED",
            or "UNKNOWN(N)" for unrecognized status codes.
        """

    @abstractmethod
    def get_sol_count(self) -> int:
        """Return the number of solutions found."""

    @abstractmethod
    def solve_stats(self) -> SolveStats:
        """Return structured solve statistics.

        Must be called after optimize(). Returns a SolveStats instance
        with fields populated for this backend; unavailable fields are None.
        """

    @abstractmethod
    def compute_iis(self) -> None:
        """Compute an Irreducible Infeasible Subsystem (IIS)."""

    @abstractmethod
    def write(self, path: str) -> None:
        """Write the model to a file (e.g. .lp, .mps, .ilp)."""

    @abstractmethod
    def quicksum(self, iterable: Any) -> LinExpr:
        """Compute the sum of an iterable of backend expressions.

        Per D-05: GurobiBackend delegates to gurobipy.quicksum.
        OR-Tools backend will use Python sum().
        """

    @abstractmethod
    def lin_expr(self, constant: float = 0.0) -> LinExpr:
        """Create a zero (or constant-valued) linear expression.

        Usable as a defaultdict value factory:
            d = defaultdict(model.lin_expr)
            d["key"] += var._raw
        """

    @property
    def supports_nonlinear(self) -> bool:
        """Whether this backend supports general non-linear constraints."""
        return False

    def add_genconstr_nl(self, resvar: SolverVar, expr: Any, *, name: str = "") -> None:
        """Add a general non-linear constraint: resvar = expr.

        Only supported by backends where supports_nonlinear is True.
        For linear-only backends, callers must use a linearized formulation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support non-linear constraints")

    def set_lexicographic_objectives(
        self,
        objectives: Sequence[ObjectiveLevel],
        *,
        sense: str = "minimize",
    ) -> None:
        """Register a hierarchy of objectives for lexicographic optimization.

        Objectives are optimized in decreasing ``priority`` order.  Each
        level is optimized subject to the constraint that higher-priority
        objectives do not degrade beyond their specified tolerances.

        Must be called before :meth:`optimize`.  Replaces any previously
        set single objective.

        **GurobiBackend** uses native multi-objective support
        (``setObjectiveN``).  **ORToolsBackend** performs sequential
        solves, locking each level with a constraint before proceeding to
        the next.

        Args:
            objectives: One or more objective levels.
            sense: ``"minimize"`` or ``"maximize"`` — applies to all levels.

        Raises:
            ValueError: If *objectives* is empty.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support lexicographic objectives")

    def infinity(self) -> float:
        """Convenience accessor for INFINITY class constant."""
        return self.INFINITY


# ---------------------------------------------------------------------------
# Gurobi private implementations
# ---------------------------------------------------------------------------


class _GurobiVar(SolverVar):
    """Gurobipy-backed SolverVar. Private — use GurobiBackend.add_var()."""

    __slots__ = ("_v",)

    def __init__(self, v: gp.Var) -> None:
        self._v = v

    @property
    def X(self) -> float:
        return self._v.X

    @property
    def _raw(self) -> gp.Var:
        """Return the underlying gp.Var for expression building."""
        return self._v

    # Arithmetic operator delegation to the underlying gp.Var.
    # Constraint expressions like `solver_var <= other_expr` remain valid
    # Gurobi TempConstr objects because we delegate to gp.Var's operators.
    def __le__(self, other: Any) -> Any:
        return self._v.__le__(_unwrap(other))

    def __ge__(self, other: Any) -> Any:
        return self._v.__ge__(_unwrap(other))

    def __eq__(self, other: Any) -> Any:
        return self._v.__eq__(_unwrap(other))

    def __ne__(self, other: Any) -> Any:
        return self._v.__ne__(_unwrap(other))

    def __add__(self, other: Any) -> Any:
        return self._v.__add__(_unwrap(other))

    def __radd__(self, other: Any) -> Any:
        return self._v.__radd__(_unwrap(other))

    def __sub__(self, other: Any) -> Any:
        return self._v.__sub__(_unwrap(other))

    def __rsub__(self, other: Any) -> Any:
        return self._v.__rsub__(_unwrap(other))

    def __mul__(self, other: Any) -> Any:
        return self._v.__mul__(_unwrap(other))

    def __rmul__(self, other: Any) -> Any:
        return self._v.__rmul__(_unwrap(other))

    def __truediv__(self, other: Any) -> Any:
        return self._v.__truediv__(_unwrap(other))

    def __rtruediv__(self, other: Any) -> Any:
        return self._v.__rtruediv__(_unwrap(other))

    def __neg__(self) -> Any:
        return self._v.__neg__()

    def __hash__(self) -> int:
        return hash(self._v)

    def __repr__(self) -> str:
        return f"_GurobiVar({self._v!r})"


class _GurobiLinExpr(LinExpr):
    """Gurobipy-backed LinExpr. Private — use GurobiBackend.lin_expr()."""

    __slots__ = ("_e",)

    def __init__(self, e: gp.LinExpr | None = None) -> None:
        self._e = e if e is not None else gp.LinExpr()

    @property
    def _raw(self) -> gp.LinExpr:
        return self._e

    def __iadd__(self, other: Any) -> _GurobiLinExpr:
        self._e += _unwrap(other)
        return self

    def __add__(self, other: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(self._e + _unwrap(other))

    def __radd__(self, other: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(_unwrap(other) + self._e)

    def __sub__(self, other: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(self._e - _unwrap(other))

    def __rsub__(self, other: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(_unwrap(other) - self._e)

    def __mul__(self, other: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(self._e * _unwrap(other))

    def __rmul__(self, other: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(_unwrap(other) * self._e)

    def __neg__(self) -> _GurobiLinExpr:
        return _GurobiLinExpr(-self._e)

    def __le__(self, other: Any) -> Any:
        return self._e.__le__(_unwrap(other))

    def __ge__(self, other: Any) -> Any:
        return self._e.__ge__(_unwrap(other))

    def __eq__(self, other: Any) -> Any:
        return self._e.__eq__(_unwrap(other))

    def __repr__(self) -> str:
        return f"_GurobiLinExpr({self._e!r})"


# ---------------------------------------------------------------------------
# GurobiBackend
# ---------------------------------------------------------------------------

# _VTYPE_MAP and _STATUS_MAP key off gurobipy's GRB constants, so they can only be built when
# gurobipy is installed. They are consumed exclusively by GurobiBackend (guarded in __init__).
_VTYPE_MAP: dict[SolverVarType, str] = {}
_STATUS_MAP: dict[int, str] = {}
if gp is not None:
    _VTYPE_MAP = {
        SolverVarType.BINARY: GRB.BINARY,
        SolverVarType.INTEGER: GRB.INTEGER,
        SolverVarType.CONTINUOUS: GRB.CONTINUOUS,
    }
    _STATUS_MAP = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }

_PARAM_MAP: dict[SolverParams, str] = {
    SolverParams.VERBOSITY: "OutputFlag",
    SolverParams.TIME_LIMIT: "TimeLimit",
    SolverParams.THREADS: "Threads",
    SolverParams.POOL_GAP: "PoolGap",
    SolverParams.LOG_TO_CONSOLE: "LogToConsole",
}


class GurobiBackend(SolverModel):
    """Gurobipy-backed SolverModel implementation.

    Per ABS-02: delegates all operations to gurobipy with zero behavioral
    change relative to the pre-abstraction code.
    """

    INFINITY: float = float("inf")  # Maps to GRB.INFINITY (also float('inf'))

    def __init__(self, name: str = "") -> None:
        if gp is None:
            raise ImportError(
                "The Gurobi backend requires gurobipy, which is not installed. Install it with "
                "`pip install 'stream-dse[gurobi]'` (a valid Gurobi license is also required at solve "
                "time). The default OR-Tools GSCIP backend needs no extra install."
            )
        self._model = gp.Model(name)

    def add_var(
        self,
        *,
        vtype: SolverVarType,
        lb: float = 0.0,
        ub: float | None = None,
        name: str = "",
    ) -> _GurobiVar:
        gv = self._model.addVar(
            vtype=_VTYPE_MAP[vtype],
            lb=lb,
            ub=ub if ub is not None else GRB.INFINITY,
            name=name,
        )
        return _GurobiVar(gv)

    def add_constr(self, expr: Any, *, name: str = "") -> None:
        self._model.addConstr(expr, name=name)

    def set_objective(self, expr: Any, *, sense: str = "minimize") -> None:
        if sense == self.MINIMIZE:
            grb_sense = GRB.MINIMIZE
        elif sense == self.MAXIMIZE:
            grb_sense = GRB.MAXIMIZE
        else:
            raise ValueError(f"Unknown objective sense: {sense!r}. Use 'minimize' or 'maximize'.")
        self._model.setObjective(expr, grb_sense)

    def set_lexicographic_objectives(
        self,
        objectives: Sequence[ObjectiveLevel],
        *,
        sense: str = "minimize",
    ) -> None:
        if not objectives:
            raise ValueError("At least one ObjectiveLevel is required")
        self._model.ModelSense = GRB.MINIMIZE if sense == self.MINIMIZE else GRB.MAXIMIZE
        for idx, obj in enumerate(sorted(objectives, key=lambda o: o.priority, reverse=True)):
            self._model.setObjectiveN(
                _unwrap(obj.expr),
                index=idx,
                priority=obj.priority,
                abstol=obj.abs_tol,
                reltol=obj.rel_tol,
                name=obj.name,
            )

    def optimize(self, callback: Any = None) -> None:
        if callback is not None:
            self._model.optimize(callback)
        else:
            self._model.optimize()

    def set_param(self, param: SolverParams, value: Any) -> None:
        if param not in _PARAM_MAP:
            raise NotImplementedError(f"Parameter {param} not supported by GurobiBackend")
        self._model.setParam(_PARAM_MAP[param], value)

    def get_status(self) -> str:
        return _STATUS_MAP.get(self._model.Status, f"UNKNOWN({self._model.Status})")

    def get_sol_count(self) -> int:
        return self._model.SolCount

    def solve_stats(self) -> SolveStats:
        has_solution = self._model.SolCount > 0
        objective: float | None = self._model.ObjVal if has_solution else None
        if has_solution:
            try:
                mip_gap: float | None = self._model.MIPGap
            except Exception:
                mip_gap = None
        else:
            mip_gap = None
        return SolveStats(
            backend="GUROBI",
            solver="gurobi",
            status=self.get_status(),
            objective=objective,
            solve_time_s=self._model.Runtime,
            mip_gap=mip_gap,
            node_count=int(self._model.NodeCount),
            iteration_count=int(self._model.IterCount),
        )

    def compute_iis(self) -> None:
        self._model.computeIIS()

    def write(self, path: str) -> None:
        self._model.write(path)

    def quicksum(self, iterable: Any) -> _GurobiLinExpr:
        return _GurobiLinExpr(gp.quicksum(_unwrap(item) for item in iterable))

    def lin_expr(self, constant: float = 0.0) -> _GurobiLinExpr:
        return _GurobiLinExpr(gp.LinExpr(constant))

    @property
    def supports_nonlinear(self) -> bool:
        return True

    def add_genconstr_nl(self, resvar: SolverVar, expr: Any, *, name: str = "") -> None:
        self._model.addGenConstrNL(resvar._raw, _unwrap(expr), name=name)

    @staticmethod
    def check_license() -> None:
        """Verify Gurobi license is available. Raises ValueError on failure."""
        try:
            tmp = gp.Model("_license_check")
            tmp.setParam("OutputFlag", 0)
            tmp.optimize()
        except gp.GurobiError as exc:
            if exc.errno == gp.GRB.Error.NO_LICENSE:
                error_message = "No valid Gurobi license found. Get an academic WLS license at https://www.gurobi.com/academia/academic-program-and-licenses/"
            else:
                error_message = f"An unexpected Gurobi error occurred: {exc.message}"
            raise ValueError(error_message) from exc


# ---------------------------------------------------------------------------
# OR-Tools MathOpt imports (Phase 2)
# ---------------------------------------------------------------------------

from ortools.math_opt.io.python import mps_converter  # noqa: E402
from ortools.math_opt.python import mathopt  # noqa: E402

# ---------------------------------------------------------------------------
# OR-Tools helpers
# ---------------------------------------------------------------------------


def _unwrap_ort(other: Any) -> Any:
    """Unwrap _ORToolsVar or _ORToolsLinExpr to underlying MathOpt object.

    Passes through mathopt.Variable, LinearSum, LinearExpression, int, float unchanged.
    """
    if isinstance(other, _ORToolsVar):
        return other._v
    if isinstance(other, _ORToolsLinExpr):
        return other._e
    return other


# ---------------------------------------------------------------------------
# OR-Tools private implementations
# ---------------------------------------------------------------------------


class _ORToolsVar(SolverVar):
    """MathOpt-backed SolverVar. Private — use ORToolsBackend.add_var()."""

    __slots__ = ("_v", "_backend")

    def __init__(self, v: mathopt.Variable, backend: ORToolsBackend) -> None:
        self._v = v
        self._backend = backend

    @property
    def X(self) -> float:
        if self._backend._result is None:
            raise ValueError("No solution available — call optimize() first")
        if not self._backend._result.has_primal_feasible_solution():
            raise ValueError("No primal feasible solution available")
        return self._backend._result.variable_values()[self._v]

    @property
    def _raw(self) -> mathopt.Variable:
        return self._v

    def __le__(self, other: Any) -> Any:
        return self._v <= _unwrap_ort(other)

    def __ge__(self, other: Any) -> Any:
        return self._v >= _unwrap_ort(other)

    def __eq__(self, other: Any) -> Any:
        return self._v == _unwrap_ort(other)

    def __ne__(self, other: Any) -> Any:
        return self._v != _unwrap_ort(other)

    def __add__(self, other: Any) -> Any:
        return self._v + _unwrap_ort(other)

    def __radd__(self, other: Any) -> Any:
        return _unwrap_ort(other) + self._v

    def __sub__(self, other: Any) -> Any:
        return self._v - _unwrap_ort(other)

    def __rsub__(self, other: Any) -> Any:
        return _unwrap_ort(other) - self._v

    def __mul__(self, other: Any) -> Any:
        return self._v * _unwrap_ort(other)

    def __rmul__(self, other: Any) -> Any:
        return _unwrap_ort(other) * self._v

    def __truediv__(self, other: Any) -> Any:
        return self._v / _unwrap_ort(other)

    def __rtruediv__(self, other: Any) -> Any:
        raise TypeError("MathOpt Variable does not support reverse division (constant / var)")

    def __neg__(self) -> Any:
        return -self._v

    def __hash__(self) -> int:
        return hash(self._v)

    def __repr__(self) -> str:
        return f"_ORToolsVar({self._v!r})"


class _ORToolsLinExpr(LinExpr):
    """MathOpt-backed LinExpr. Private — use ORToolsBackend.lin_expr()."""

    __slots__ = ("_e",)

    def __init__(self, e: Any = None) -> None:
        self._e = e if e is not None else 0

    @property
    def _raw(self) -> Any:
        return self._e

    def __iadd__(self, other: Any) -> _ORToolsLinExpr:
        self._e = self._e + _unwrap_ort(other)
        return self

    def __add__(self, other: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(self._e + _unwrap_ort(other))

    def __radd__(self, other: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(_unwrap_ort(other) + self._e)

    def __sub__(self, other: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(self._e - _unwrap_ort(other))

    def __rsub__(self, other: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(_unwrap_ort(other) - self._e)

    def __mul__(self, other: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(self._e * _unwrap_ort(other))

    def __rmul__(self, other: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(_unwrap_ort(other) * self._e)

    def __neg__(self) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(-self._e)

    def __le__(self, other: Any) -> Any:
        return self._e <= _unwrap_ort(other)

    def __ge__(self, other: Any) -> Any:
        return self._e >= _unwrap_ort(other)

    def __eq__(self, other: Any) -> Any:
        return self._e == _unwrap_ort(other)

    def __repr__(self) -> str:
        return f"_ORToolsLinExpr({self._e!r})"


# ---------------------------------------------------------------------------
# OR-Tools status and parameter maps
# ---------------------------------------------------------------------------

_ORT_STATUS_MAP: dict[mathopt.TerminationReason, str] = {
    mathopt.TerminationReason.OPTIMAL: "OPTIMAL",
    mathopt.TerminationReason.INFEASIBLE: "INFEASIBLE",
    mathopt.TerminationReason.NO_SOLUTION_FOUND: "TIME_LIMIT",
    mathopt.TerminationReason.FEASIBLE: "TIME_LIMIT",
    mathopt.TerminationReason.UNBOUNDED: "UNBOUNDED",
    mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED: "INF_OR_UNBD",
    mathopt.TerminationReason.NUMERICAL_ERROR: "UNKNOWN(NUMERICAL_ERROR)",
    mathopt.TerminationReason.OTHER_ERROR: "UNKNOWN(OTHER_ERROR)",
}


def _to_timedelta(v: Any) -> datetime.timedelta:
    return datetime.timedelta(seconds=v)


_ORT_PARAM_MAP: dict[SolverParams, tuple[str, Any]] = {
    SolverParams.VERBOSITY: ("enable_output", bool),
    SolverParams.TIME_LIMIT: ("time_limit", _to_timedelta),
    SolverParams.THREADS: ("threads", int),
    SolverParams.POOL_GAP: ("relative_gap_tolerance", float),
    SolverParams.LOG_TO_CONSOLE: ("enable_output", bool),
}


# ---------------------------------------------------------------------------
# ORToolsBackend
# ---------------------------------------------------------------------------


class ORToolsBackend(SolverModel):
    """OR-Tools MathOpt-backed SolverModel implementation (Phase 2).

    Uses MathOpt API for backend-agnostic MILP solving. Supports binary,
    integer, and continuous variables; linear constraints; objective
    minimization/maximization; and MPS export on infeasibility.

    Default solver: GSCIP (bundled in pip, full MILP support).
    """

    INFINITY: float = math.inf

    def __init__(self, name: str = "", solver_type: mathopt.SolverType = mathopt.SolverType.GSCIP) -> None:
        self._model = mathopt.Model(name=name)
        self._solver_type = solver_type
        self._params: dict[str, Any] = {"enable_output": False}
        self._result: mathopt.SolveResult | None = None
        # MathOpt requires globally-unique names within a model.  These counters
        # deduplicate names by appending a suffix when the same base name would
        # be used a second time (Gurobi silently allows duplicate names; MathOpt does not).
        self._var_name_count: dict[str, int] = {}
        self._constr_name_count: dict[str, int] = {}
        # Lexicographic objective state (set by set_lexicographic_objectives)
        self._lex_objectives: list[ObjectiveLevel] | None = None
        self._lex_sense: str = self.MINIMIZE

    def _unique_var_name(self, name: str) -> str:
        """Return a unique variable name by appending a counter on collision."""
        count = self._var_name_count.get(name, 0)
        self._var_name_count[name] = count + 1
        return name if count == 0 else f"{name}_{count}"

    def _unique_constr_name(self, name: str) -> str:
        """Return a unique constraint name by appending a counter on collision."""
        count = self._constr_name_count.get(name, 0)
        self._constr_name_count[name] = count + 1
        return name if count == 0 else f"{name}_{count}"

    def add_var(
        self,
        *,
        vtype: SolverVarType,
        lb: float = 0.0,
        ub: float | None = None,
        name: str = "",
    ) -> _ORToolsVar:
        ub_val = math.inf if ub is None else ub
        unique_name = self._unique_var_name(name)
        if vtype == SolverVarType.BINARY:
            v = self._model.add_binary_variable(name=unique_name)
        elif vtype == SolverVarType.INTEGER:
            v = self._model.add_integer_variable(lb=lb, ub=ub_val, name=unique_name)
        else:
            v = self._model.add_variable(lb=lb, ub=ub_val, name=unique_name)
        return _ORToolsVar(v, self)

    def add_constr(self, expr: Any, *, name: str = "") -> None:
        unique_name = self._unique_constr_name(name)
        self._model.add_linear_constraint(_unwrap_ort(expr), name=unique_name)

    def set_objective(self, expr: Any, *, sense: str = "minimize") -> None:
        raw_expr = _unwrap_ort(expr)
        if sense == self.MINIMIZE:
            self._model.minimize(raw_expr)
        elif sense == self.MAXIMIZE:
            self._model.maximize(raw_expr)
        else:
            raise ValueError(f"Unknown objective sense: {sense!r}. Use 'minimize' or 'maximize'.")

    def set_lexicographic_objectives(
        self,
        objectives: Sequence[ObjectiveLevel],
        *,
        sense: str = "minimize",
    ) -> None:
        if not objectives:
            raise ValueError("At least one ObjectiveLevel is required")
        self._lex_objectives = sorted(objectives, key=lambda o: o.priority, reverse=True)
        self._lex_sense = sense

    def optimize(self, callback: Any = None) -> None:
        if self._lex_objectives is not None:
            self._optimize_lexicographic()
        else:
            params = mathopt.SolveParameters(**self._params)
            self._result = mathopt.solve(self._model, self._solver_type, params=params)

    def _optimize_lexicographic(self) -> None:
        assert self._lex_objectives is not None
        params = mathopt.SolveParameters(**self._params)

        for i, obj in enumerate(self._lex_objectives):
            raw_expr = _unwrap_ort(obj.expr)
            if self._lex_sense == self.MINIMIZE:
                self._model.minimize(raw_expr)
            else:
                self._model.maximize(raw_expr)

            self._result = mathopt.solve(self._model, self._solver_type, params=params)

            if self._result is None or not self._result.has_primal_feasible_solution():
                _logger.warning("Lexicographic solve: phase %d (%s) found no feasible solution", i, obj.name)
                return

            if i < len(self._lex_objectives) - 1:
                opt_val = self._result.objective_value()
                tol = obj.abs_tol + abs(opt_val) * obj.rel_tol
                if self._lex_sense == self.MINIMIZE:
                    self.add_constr(raw_expr <= opt_val + tol, name=f"_lex_lock_{obj.name}_{i}")
                else:
                    self.add_constr(raw_expr >= opt_val - tol, name=f"_lex_lock_{obj.name}_{i}")

    def set_param(self, param: SolverParams, value: Any) -> None:
        if param not in _ORT_PARAM_MAP:
            raise NotImplementedError(f"Parameter {param} not supported by ORToolsBackend")
        attr_name, converter = _ORT_PARAM_MAP[param]
        self._params[attr_name] = converter(value)

    def get_status(self) -> str:
        if self._result is None:
            return "UNKNOWN(None)"
        return _ORT_STATUS_MAP.get(self._result.termination.reason, f"UNKNOWN({self._result.termination.reason})")

    def get_sol_count(self) -> int:
        if self._result is None:
            return 0
        return 1 if self._result.has_primal_feasible_solution() else 0

    def solve_stats(self) -> SolveStats:
        has_solution = self._result is not None and self._result.has_primal_feasible_solution()
        objective: float | None = self._result.objective_value() if has_solution else None
        if self._result is not None:
            solve_time_s = self._result.solve_stats.solve_time.total_seconds()
        else:
            solve_time_s = 0.0
        return SolveStats(
            backend=f"ORTOOLS_{self._solver_type.name}",
            solver=self._solver_type.name.lower(),
            status=self.get_status(),
            objective=objective,
            solve_time_s=solve_time_s,
            mip_gap=None,
            node_count=None,
            iteration_count=None,
        )

    def compute_iis(self) -> None:
        _logger.warning(
            "ORToolsBackend does not support IIS computation. Use write() to export the model as MPS for debugging."
        )

    def write(self, path: str) -> None:
        actual_path = path
        if path.endswith(".ilp"):
            actual_path = path[:-4] + ".mps"
            _logger.info("ORToolsBackend exports MPS format only. Writing to %s instead of %s", actual_path, path)
        proto = self._model.export_model()
        mps_str = mps_converter.model_proto_to_mps(proto)
        with open(actual_path, "w") as f:
            f.write(mps_str)

    def quicksum(self, iterable: Any) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(mathopt.fast_sum(_unwrap_ort(item) for item in iterable))

    def lin_expr(self, constant: float = 0.0) -> _ORToolsLinExpr:
        return _ORToolsLinExpr(constant)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


_ORTOOLS_SOLVER_MAP: dict[SolverBackend, Any] = {
    SolverBackend.ORTOOLS_GSCIP: mathopt.SolverType.GSCIP,
    SolverBackend.ORTOOLS_HIGHS: mathopt.SolverType.HIGHS,
    SolverBackend.ORTOOLS_GUROBI: mathopt.SolverType.GUROBI,
}


def create_solver(backend: SolverBackend, name: str = "") -> SolverModel:
    """Instantiate a SolverModel by backend enum.

    Each SolverBackend member maps to exactly one solver configuration —
    no additional solver_type kwarg needed.

    Args:
        backend: The solver backend to use.
        name: Model name (used in logs and output files).

    Returns:
        A concrete SolverModel instance.

    Raises:
        ValueError: For unrecognized backend values.
    """
    if backend == SolverBackend.GUROBI:
        return GurobiBackend(name)
    if backend in _ORTOOLS_SOLVER_MAP:
        return ORToolsBackend(name, _ORTOOLS_SOLVER_MAP[backend])
    raise ValueError(f"Unknown backend: {backend!r}")
