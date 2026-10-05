"""The files a solve writes beside its result, unless ``SolveOptions(artifacts=False)``: the solver's metrics and
progress and the slot latency breakdown (``reports/``), the Perfetto traces of the schedule (``traces/``), and the
solver's progress plot and a picture of the solved workload (``figures/``)."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import yaml

# gurobipy is optional: GRB supplies the callback codes of the solver progress and the Gurobi status names of the
# metrics, and only a Gurobi solve reaches them.
try:
    from gurobipy import GRB
except ModuleNotFoundError:
    GRB = None  # type: ignore[assignment]

from stream.opt.allocation.constraint_optimization.report import slot_latency_breakdown
from stream.profiling import span
from stream.visualization.steady_state_trace import export_steady_state_trace

if TYPE_CHECKING:
    from stream.allocation.schedule import SteadyStateSchedule
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.opt.solver import SolverModel

logger = logging.getLogger(__name__)

Trace = list[dict[str, Any]]


class SolveProgress:
    """A Gurobi callback recording the incumbent, bound and effort of a solve as it runs, for its progress plot,
    trace and metrics."""

    def __init__(self) -> None:
        self.trace: Trace = []

    def __call__(self, model: Any, where: int) -> None:
        if where not in (GRB.Callback.MIP, GRB.Callback.MIPSOL, GRB.Callback.PRESOLVE):
            return

        if where == GRB.Callback.PRESOLVE:
            point = {
                "event": "PRESOLVE",
                "time": float(model.cbGet(GRB.Callback.RUNTIME)),
                "work": float(model.cbGet(GRB.Callback.WORK)),
                "rows_removed": int(model.cbGet(GRB.Callback.PRE_ROWDEL)),
                "cols_removed": int(model.cbGet(GRB.Callback.PRE_COLDEL)),
                "bound_changes": int(model.cbGet(GRB.Callback.PRE_BNDCHG)),
                "coeff_changes": int(model.cbGet(GRB.Callback.PRE_COECHG)),
            }
            self.trace.append(point)
            return

        if where == GRB.Callback.MIP:
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
            nodlft = model.cbGet(GRB.Callback.MIP_NODLFT)
            itrcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            work = model.cbGet(GRB.Callback.WORK)
            event = "MIP"
        else:
            best = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
            nodlft = None
            itrcnt = None
            cutcnt = None
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            work = model.cbGet(GRB.Callback.WORK)
            event = "MIPSOL"

        max_val = 1e90
        if not math.isfinite(best) or abs(best) >= max_val:
            best = None
        else:
            best = float(best)

        if not math.isfinite(bound) or abs(bound) >= max_val:
            bound = None
        else:
            bound = float(bound)

        gap = None
        if best is not None and bound is not None:
            gap = abs(best - bound) / max(1.0, abs(best))

        point = {
            "event": event,
            "time": float(runtime),
            "work": float(work),
            "nodecnt": float(nodecnt) if nodecnt is not None else None,
            "nodlft": float(nodlft) if nodlft is not None else None,
            "itrcnt": float(itrcnt) if itrcnt is not None else None,
            "cutcnt": int(cutcnt) if cutcnt is not None else None,
            "best_obj": best,
            "best_bound": bound,
            "gap": gap,
        }

        self.trace.append(point)


def write_artifacts(
    directory: str, allocator: TransferAndTensorAllocator, schedule: SteadyStateSchedule, progress: SolveProgress
) -> None:
    """Write the artifacts of a solved allocation under ``directory``; an artifact that cannot be written is logged
    and left out, never failing the solve."""
    reports, traces, figures = (os.path.join(directory, kind) for kind in ("reports", "traces", "figures"))
    for path in (reports, traces, figures):
        os.makedirs(path, exist_ok=True)
    with span("artifact_reports"):
        _observe("optimization trace", save_optimization_trace, progress.trace, f"{reports}/optimization_trace.yaml")
        _observe(
            "optimization metrics",
            save_optimization_metrics,
            allocator.model,
            progress.trace,
            f"{reports}/optimization_metrics.yaml",
        )
        _observe(
            "slot latency breakdown",
            _save_slot_latency_breakdown,
            allocator,
            schedule,
            f"{reports}/slot_latency_breakdown.yaml",
        )
    with span("artifact_traces"):
        latency = schedule.solution.latency
        for compact, fname in [(True, "steady_state_trace_compact.json"), (False, "steady_state_trace.json")]:
            _observe(
                f"steady-state trace {fname}",
                export_steady_state_trace,
                allocator.context,
                schedule.solution.transfer_routes,
                iterations=schedule.iterations,
                overlap=latency.overlap,
                latency_per_iteration=latency.per_iteration,
                output_path=traces,
                compact=compact,
                filename=fname,
            )
    with span("artifact_figures"):
        _observe(
            "optimization progress plot",
            plot_optimization_progress,
            progress.trace,
            show=False,
            save_path=f"{figures}/optimization_progress.png",
        )
        _observe(
            "solved workload",
            schedule.workload.visualize,
            f"{figures}/steady_state_workload_final.svg",
            schedule.mapping,
            schedule.ssis,
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


def _save_slot_latency_breakdown(
    allocator: TransferAndTensorAllocator, schedule: SteadyStateSchedule, save_path: str
) -> None:
    """The slot latency breakdown of the solved model as YAML, next to the metrics."""
    breakdown = slot_latency_breakdown(allocator.context, dict(schedule.solution.reuse_levels))
    with open(save_path, "w") as fh:
        yaml.safe_dump(breakdown, fh, sort_keys=False, default_flow_style=False)
    logger.info("Slot latency breakdown saved to %s", save_path)


def save_optimization_metrics(model: SolverModel, trace: Trace, save_path: str) -> None:
    """
    Dump a concise YAML summary of the Gurobi run alongside the progress plot.

    Headline fields (in order of relevance for a paper):
      - search:    nodes explored, simplex/barrier iterations  -> "how much was searched"
      - solution:  objective, best bound, MIP gap              -> "what was found / how tight"
      - effort:    runtime (s), work units                     -> "how expensive it was"
      - model:    variable / constraint / nonzero counts      -> "problem size"
      - trace:     per-event progress records
    """

    def _attr(name: str) -> Any | None:
        """Return a Gurobi model attribute or None if unavailable post-solve."""
        # Access the underlying gurobipy model for Gurobi-specific attributes.
        # GurobiBackend stores the model as ._model; fall back gracefully for other backends.
        from stream.opt.solver import GurobiBackend  # noqa: PLC0415

        raw_model = model._model if isinstance(model, GurobiBackend) else None
        if raw_model is None:
            return None
        try:
            value = getattr(raw_model, name)
        except Exception:  # noqa: BLE001  # catches AttributeError and gp.GurobiError
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    status = _attr("Status")
    # The status code is a Gurobi attribute (None for other backends). Only translate it via
    # GRB constants when gurobipy is installed; otherwise it is already None.
    if GRB is not None:
        status_name = {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.WORK_LIMIT: "WORK_LIMIT",
        }.get(status, str(status))
    else:
        status_name = str(status)

    # Per-event trace, normalized to a compact form (drop None values)
    trace_records: list[dict[str, Any]] = []
    for rec in trace:
        entry: dict[str, Any] = {"time_s": rec.get("time"), "event": rec.get("event")}
        for src, dst in (
            ("best_obj", "incumbent"),
            ("best_bound", "best_bound"),
            ("gap", "gap"),
            ("nodecnt", "nodes"),
            ("cutcnt", "cuts"),
            ("work", "work"),
        ):
            val = rec.get(src)
            if isinstance(val, int | float) and math.isfinite(val):
                entry[dst] = val
        trace_records.append(entry)

    metrics: dict[str, Any] = {
        "status": status_name,
        "search": {
            "nodes_explored": _attr("NodeCount"),
            "simplex_iterations": _attr("IterCount"),
            "barrier_iterations": _attr("BarIterCount"),
        },
        "solution": {
            "objective": _attr("ObjVal"),
            "best_bound": _attr("ObjBound"),
            "mip_gap": _attr("MIPGap"),
        },
        "effort": {
            "runtime_s": _attr("Runtime"),
            "work_units": _attr("Work"),
        },
        "model": {
            "variables": {
                "total": _attr("NumVars"),
                "integer": _attr("NumIntVars"),
                "binary": _attr("NumBinVars"),
            },
            "constraints": {
                "linear": _attr("NumConstrs"),
                "general": _attr("NumGenConstrs"),
            },
            "nonzeros": _attr("NumNZs"),
        },
        "trace": trace_records,
    }

    with open(save_path, "w") as fh:
        yaml.safe_dump(metrics, fh, sort_keys=False, default_flow_style=False)
    logger.info("Optimization metrics saved to %s", save_path)


def plot_optimization_progress(  # noqa: PLR0912, PLR0915
    trace: Trace,
    *,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10.0, 6),
    show_work_subplot: bool = False,
) -> None:
    """
    Plot the optimization progress a :class:`SolveProgress` recorded.

    Top subplot:
    - best incumbent
    - best bound
    - relative gap (%) on a secondary y-axis

    Bottom subplot (optional):
    - solver work (if available)
    - optionally explored nodes / cuts on a secondary y-axis

    Args:
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the figure.
        figsize: Figure size as (width, height).
        show_work_subplot: Whether to include the bottom work subplot.

    Expects callback records like:
        {
            "event": "MIP" or "MIPSOL",
            "time": float,
            "nodecnt": float | None,
            "best_obj": float | None,
            "best_bound": float | None,
            "gap": float | None,
            "work": float | None,
            "cutcnt": float | None,
        }
    """

    if not trace:
        # Non-Gurobi backends record no progress.
        return

    def _is_finite_number(x) -> bool:
        return x is not None and isinstance(x, int | float) and math.isfinite(x)

    trace = [
        rec
        for rec in trace
        if _is_finite_number(rec.get("time"))
        and (
            _is_finite_number(rec.get("best_obj"))
            or _is_finite_number(rec.get("best_bound"))
            or _is_finite_number(rec.get("gap"))
            or _is_finite_number(rec.get("work"))
            or _is_finite_number(rec.get("nodecnt"))
            or _is_finite_number(rec.get("cutcnt"))
        )
    ]

    if not trace:
        raise ValueError("Optimization trace exists, but it does not contain plottable finite values.")

    trace.sort(key=lambda r: (float(r["time"]), 0 if r.get("event") == "MIPSOL" else 1))

    times: list[float] = []
    incumbent: list[float] = []
    bound: list[float] = []
    gap_pct: list[float] = []

    works: list[float] = []
    nodes: list[float] = []
    cuts: list[float] = []

    last_best_obj: float | None = None
    last_best_bound: float | None = None
    last_gap: float | None = None
    last_work: float | None = None
    last_nodecnt: float | None = None
    last_cutcnt: float | None = None

    for rec in trace:
        t = float(rec["time"])

        if _is_finite_number(rec.get("best_obj")):
            last_best_obj = float(rec["best_obj"])
        if _is_finite_number(rec.get("best_bound")):
            last_best_bound = float(rec["best_bound"])
        if _is_finite_number(rec.get("gap")):
            last_gap = 100.0 * float(rec["gap"])

        if _is_finite_number(rec.get("work")):
            last_work = float(rec["work"])
        if _is_finite_number(rec.get("nodecnt")):
            last_nodecnt = float(rec["nodecnt"])
        if _is_finite_number(rec.get("cutcnt")):
            last_cutcnt = float(rec["cutcnt"])

        times.append(t)
        incumbent.append(float("nan") if last_best_obj is None else last_best_obj)
        bound.append(float("nan") if last_best_bound is None else last_best_bound)
        gap_pct.append(float("nan") if last_gap is None else last_gap)

        works.append(float("nan") if last_work is None else last_work)
        nodes.append(float("nan") if last_nodecnt is None else last_nodecnt)
        cuts.append(float("nan") if last_cutcnt is None else last_cutcnt)

    num_subplots = 2 if show_work_subplot else 1
    height_ratios = [2.0, 1.2] if show_work_subplot else [1.0]

    fig, axes = plt.subplots(
        num_subplots,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    if num_subplots == 1:
        ax1 = axes
        ax3 = None
    else:
        ax1, ax3 = axes

    # Top subplot: original functionality unchanged
    ax2 = ax1.twinx()

    line_inc = ax1.step(times, incumbent, where="post", label="Best incumbent")
    line_bnd = ax1.step(times, bound, where="post", label="Best bound")
    line_gap = ax2.plot(times, gap_pct, label="Gap (%)", linestyle="--")

    ax1.set_ylabel("Objective")
    ax2.set_ylabel("Gap (%)")
    ax1.set_title("Gurobi optimization progress")
    ax1.grid(True, alpha=0.3)

    handles_top = line_inc + line_bnd + line_gap
    labels_top = [h.get_label() for h in handles_top]
    ax1.legend(handles_top, labels_top, loc="best")

    # Bottom subplot: work done (optional)
    if show_work_subplot:
        ax4 = ax3.twinx()
        handles_bottom = []

        if any(not math.isnan(x) for x in works):
            line_work = ax3.step(times, works, where="post", label="Solver work")
            handles_bottom += line_work

        if any(not math.isnan(x) for x in nodes):
            line_nodes = ax4.step(times, nodes, where="post", label="Explored nodes", linestyle="--")
            handles_bottom += line_nodes

        if any(not math.isnan(x) for x in cuts):
            line_cuts = ax4.step(times, cuts, where="post", label="Cuts applied", linestyle=":")
            handles_bottom += line_cuts

        ax3.set_xlabel("Runtime (s)")
        ax3.set_ylabel("Work units")
        ax4.set_ylabel("Nodes / cuts")
        ax3.grid(True, alpha=0.3)

        if handles_bottom:
            labels_bottom = [h.get_label() for h in handles_bottom]
            ax3.legend(handles_bottom, labels_bottom, loc="best")
    else:
        ax1.set_xlabel("Runtime (s)")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Optimization progress plot saved to %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_optimization_trace(trace: Trace, file_path: str) -> None:
    """
    Save the optimization trace to a YAML file.

    Writes a single chronological ``trace`` list. Each entry represents a
    point where something changed and has the following fields:

    - ``time_s``     – solver runtime in seconds
    - ``event``      – ``"MIPSOL"`` (new incumbent found) or ``"MIP"`` (bound update)
    - ``incumbent``  – present only on ``MIPSOL`` entries (when best_obj improved)
    - ``best_bound`` – present only when the bound changed
    - ``gap``        – relative gap at this point (when both values are available)

    Args:
        file_path: Destination path for the YAML file (e.g. "trace.yaml").
    """
    if not trace:
        # Non-Gurobi backends record no progress.
        return

    def _fin(x) -> bool:
        return x is not None and isinstance(x, int | float) and math.isfinite(x)

    # Sort by time, MIPSOL first when times are equal (mirrors plot logic)
    sorted_trace = sorted(
        trace,
        key=lambda r: (float(r["time"]), 0 if r.get("event") == "MIPSOL" else 1),
    )

    entries: list[dict] = []
    last_obj: float | None = None
    last_bound: float | None = None

    for rec in sorted_trace:
        if not _fin(rec.get("time")):
            continue

        t = float(rec["time"])
        obj = float(rec["best_obj"]) if _fin(rec.get("best_obj")) else None
        bnd = float(rec["best_bound"]) if _fin(rec.get("best_bound")) else None
        gap = float(rec["gap"]) if _fin(rec.get("gap")) else None

        incumbent_improved = obj is not None and obj != last_obj
        bound_changed = bnd is not None and bnd != last_bound

        if not incumbent_improved and not bound_changed:
            continue

        entry: dict = {"time_s": t, "event": rec.get("event", "MIP")}
        if incumbent_improved:
            entry["incumbent"] = obj
            last_obj = obj
        if bound_changed:
            entry["best_bound"] = bnd
            last_bound = bnd
        if gap is not None:
            entry["gap"] = gap

        entries.append(entry)

    with open(file_path, "w") as f:
        yaml.dump({"trace": entries}, f, default_flow_style=False, sort_keys=False)

    logger.info("Optimization trace saved to %s", file_path)
