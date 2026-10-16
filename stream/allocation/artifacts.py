"""The files a solve writes beside its result, unless ``SolveOptions(artifacts=False)``: the solver's metrics and
progress and the slot latency breakdown (``reports/``), the Perfetto traces of the allocation (``traces/``), and the
solver's progress plot and a picture of the solved workload (``figures/``)."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import yaml

try:
    from gurobipy import GRB
except ModuleNotFoundError:
    GRB = None  # type: ignore[assignment]

from stream.profiling import span
from stream.visualization.steady_state_trace import export_steady_state_trace

if TYPE_CHECKING:
    from stream.allocation.allocation import Allocation
    from stream.opt.solver import SolverModel

logger = logging.getLogger(__name__)

_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

TRACE_KEYS = (
    "time_s",
    "event",
    "incumbent_objective",
    "objective_bound",
    "mip_gap",
    "nodes_explored",
    "nodes_left",
    "simplex_iterations",
    "cuts_applied",
    "work_units",
    "rows_removed",
    "columns_removed",
    "bound_changes",
    "coefficient_changes",
)
"""The keys of each point of a solve's progress, in the order its trace writes them."""

NO_VALUE = 1e90
"""Gurobi's stand-in for an incumbent or bound a callback does not have yet."""


class SolveProgress:
    """A Gurobi callback recording each presolve, search and new-incumbent point of a solve as it runs, for its trace
    and progress plot; a backend without callbacks records none."""

    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []

    def __call__(self, model: Any, where: int) -> None:
        cb = GRB.Callback
        if where == cb.PRESOLVE:
            point: dict[str, Any] = {
                "event": "PRESOLVE",
                "rows_removed": int(model.cbGet(cb.PRE_ROWDEL)),
                "columns_removed": int(model.cbGet(cb.PRE_COLDEL)),
                "bound_changes": int(model.cbGet(cb.PRE_BNDCHG)),
                "coefficient_changes": int(model.cbGet(cb.PRE_COECHG)),
            }
        elif where == cb.MIP:
            point = {
                "event": "MIP",
                "incumbent_objective": _finite(model.cbGet(cb.MIP_OBJBST)),
                "objective_bound": _finite(model.cbGet(cb.MIP_OBJBND)),
                "nodes_explored": int(model.cbGet(cb.MIP_NODCNT)),
                "nodes_left": int(model.cbGet(cb.MIP_NODLFT)),
                "simplex_iterations": int(model.cbGet(cb.MIP_ITRCNT)),
                "cuts_applied": int(model.cbGet(cb.MIP_CUTCNT)),
            }
        elif where == cb.MIPSOL:
            point = {
                "event": "MIPSOL",
                "incumbent_objective": _finite(model.cbGet(cb.MIPSOL_OBJ)),
                "objective_bound": _finite(model.cbGet(cb.MIPSOL_OBJBND)),
                "nodes_explored": int(model.cbGet(cb.MIPSOL_NODCNT)),
            }
        else:
            return
        point["time_s"] = float(model.cbGet(cb.RUNTIME))
        point["work_units"] = float(model.cbGet(cb.WORK))
        best, bound = point.get("incumbent_objective"), point.get("objective_bound")
        if best is not None and bound is not None:
            point["mip_gap"] = abs(best - bound) / max(1.0, abs(best))
        self.points.append({key: point.get(key) for key in TRACE_KEYS})

    def trace(self) -> list[dict[str, Any]]:
        """The points in time order, a new incumbent before the search point of the same time."""
        return sorted(self.points, key=lambda p: (p["time_s"], 0 if p["event"] == "MIPSOL" else 1))


def _finite(value: float) -> float | None:
    """A callback value, or None for :data:`NO_VALUE`."""
    return float(value) if math.isfinite(value) and abs(value) < NO_VALUE else None


def write_artifacts(directory: str, allocation: Allocation, progress: SolveProgress) -> None:
    """Write the artifacts of a solved allocation under ``directory``; an artifact that cannot be written is logged
    and left out, never failing the solve."""
    reports, traces, figures = (os.path.join(directory, kind) for kind in ("reports", "traces", "figures"))
    for path in (reports, traces, figures):
        os.makedirs(path, exist_ok=True)
    solution, trace = allocation.solution, progress.trace()
    with span("artifact_reports"):
        if trace:
            _observe("optimization trace", _write_yaml, {"trace": trace}, f"{reports}/optimization_trace.yaml")
        _observe("optimization metrics", _write_yaml, solution.metrics, f"{reports}/optimization_metrics.yaml")
        if solution.slot_latency_breakdown is not None:
            breakdown = solution.slot_latency_breakdown
            _observe("slot latency breakdown", _write_yaml, breakdown, f"{reports}/slot_latency_breakdown.yaml")
    with span("artifact_traces"):
        for compact, filename in [(True, "steady_state_trace_compact.json"), (False, "steady_state_trace.json")]:
            _observe(
                f"steady-state trace {filename}",
                export_steady_state_trace,
                allocation,
                traces,
                compact=compact,
                filename=filename,
            )
    with span("artifact_figures"):
        if trace:
            _observe(
                "optimization progress plot", plot_optimization_progress, trace, f"{figures}/optimization_progress.png"
            )
        _observe(
            "solved workload",
            allocation.problem.workload.visualize,
            f"{figures}/steady_state_workload_final.svg",
            allocation.mapping,
            allocation.ssis,
        )


def write_infeasible_model(model: SolverModel, output_path: str) -> None:
    """Write an infeasible model for offline debugging: Gurobi's IIS as ``model.ilp``, OR-Tools' model as MPS."""
    _observe("infeasible model", _write_model, model, output_path)


def _write_model(model: SolverModel, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    model.write(os.path.join(output_path, "model.ilp"))


def _observe(what: str, write: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Write one artifact; a failure is logged, an artifact being optional to the solve it observes."""
    try:
        write(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- an artifact must not fail the solve it observes
        logger.warning("Failed to write the %s: %s", what, exc)


def _write_yaml(data: dict[str, Any], path: str) -> None:
    with open(path, "w") as fh:
        yaml.dump(data, fh, Dumper=_DUMPER, sort_keys=False, default_flow_style=False)
    logger.info("Saved %s", path)


def plot_optimization_progress(trace: list[dict[str, Any]], save_path: str) -> None:
    """The best incumbent, the best bound and the gap over the runtime of a solve, from its trace, as a PNG."""
    series: dict[str, list[float]] = {key: [] for key in ("incumbent_objective", "objective_bound", "mip_gap")}
    current = dict.fromkeys(series, math.nan)
    for point in trace:
        for key, values in series.items():
            if point[key] is not None:
                current[key] = point[key]
            values.append(current[key])
    times = [point["time_s"] for point in trace]
    # matplotlib takes a third of a second to import, which a run that plots nothing should not pay
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax1 = plt.subplots(figsize=(10.0, 6))
    ax2 = ax1.twinx()
    lines = ax1.step(times, series["incumbent_objective"], where="post", label="Best incumbent")
    lines += ax1.step(times, series["objective_bound"], where="post", label="Best bound")
    lines += ax2.plot(times, [100.0 * gap for gap in series["mip_gap"]], label="Gap (%)", linestyle="--")
    ax1.set_xlabel("Runtime (s)")
    ax1.set_ylabel("Objective")
    ax2.set_ylabel("Gap (%)")
    ax1.set_title("Gurobi optimization progress")
    ax1.grid(True, alpha=0.3)
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
