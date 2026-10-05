"""The performance report and capacity slack of a solved allocation model, read off its solution once."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from stream.allocation.solution import end_to_end_mac_utilization
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.families import ReportingFamily
from stream.opt.allocation.constraint_optimization.families.latency import node_runtime
from stream.opt.allocation.constraint_optimization.families.memory import MEMORY_CAPACITY
from stream.opt.allocation.constraint_optimization.utils import active_fraction, get_active_latency
from stream.workload.utils import is_mac_operator_type
from stream.workload.workload import Tensor

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.families import FamilySelection
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.solver import SolverModel

logger = logging.getLogger(__name__)

VAR_THRESHOLD = 0.5
"""A MILP binary comes back as 0.9999...; anything above the midpoint is a 1."""

_OCCUPANCY_TOP_TENSORS = 8


def guarded[T](section: str, compute: Callable[..., T], *args: Any) -> T | None:
    """``compute(*args)``, or None with a warning when it fails: a report must not fail the solve it reports on."""
    try:
        return compute(*args)
    except Exception as exc:  # noqa: BLE001 -- a report must not fail the solve it reports on
        logger.warning("Could not compute the %s: %s", section, exc)
        return None


def solved_reports(
    ctx: FormulationContext,
    families: FamilySelection,
    reuse_levels: dict[Tensor, int],
    total_latency: int,
    total_mac_ops: int | None,
) -> tuple[dict[str, Any] | None, dict[int, dict[str, float]] | None, dict[str, Any] | None]:
    """The performance report of the solved allocation -- where its latency goes, what binds its overlap, what each
    tensor and memory holds, and the families' sections --, the unused capacity per core and the slot latency
    breakdown, each read off the solution once and None when it cannot be."""
    reuse = guarded("tensor reuse", tensor_reuse_breakdown, ctx, reuse_levels)
    slack = guarded("resource slack", resource_slack, ctx)
    occupancy = guarded("memory occupancy", memory_occupancy, ctx)
    performance = guarded(
        "performance report", _performance, ctx, families, total_mac_ops, total_latency, reuse, slack, occupancy
    )
    return (
        performance,
        guarded("capacity slack", capacity_slack, ctx, occupancy),
        guarded("slot latency breakdown", slot_latency_breakdown, ctx, reuse, slack),
    )


def _performance(
    ctx: FormulationContext,
    families: FamilySelection,
    total_mac_ops: int | None,
    total_latency: int,
    reuse: list[dict[str, Any]] | None,
    slack: list[dict[str, Any]] | None,
    occupancy: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    performance = {
        **_utilization(ctx, total_mac_ops, total_latency),
        "overlap": overlap_section(ctx, slack),
        "tensor_reuse": reuse,
        "memory_occupancy": occupancy,
    }
    for family in families.families:
        if isinstance(family, ReportingFamily):
            performance |= guarded(f"{family.name} report", family.report, ctx) or {family.name: None}
    return performance


def _utilization(ctx: FormulationContext, total_mac_ops: int | None, total_latency: int) -> dict[str, Any]:
    """Per compute node its cores, latency, MAC utilization and whether it fell back to a scalar estimate; the
    per-iteration latency split into compute- and transfer-bound slots; and the aggregate utilization."""
    space = ctx.space
    lut = space.cost_lut
    per_node: dict[str, dict[str, Any]] = {}
    compute_by_slot: dict[int, float] = {}
    for n in space.ssc_nodes:
        cores = lut.get_cores(n)
        active = get_active_latency(n, float(node_runtime(space, n)), space.ssis)
        s = space.slot_of[n]
        compute_by_slot[s] = max(compute_by_slot.get(s, 0.0), float(active))
        if not cores:
            continue
        entry = lut.get_cost(n, cores[0])
        ideal = getattr(entry, "ideal_cycle", None)
        mac_util = getattr(entry, "mac_spatial_utilization", None)
        node_type = getattr(getattr(entry, "layer", None), "type", "")
        per_node[n.name] = {
            "kind": "compute",
            "n_cores": len(cores),
            "latency_cycles": int(active),
            "ideal_compute_cycles": _json_scalar(ideal),
            "mac_spatial_utilization": _json_scalar(mac_util),
            "compute_efficiency": _json_scalar(float(ideal) / active if ideal and active else None),
            "fallback": getattr(entry, "cme", None) is None and is_mac_operator_type(node_type),
        }

    compute_cycles = transfer_cycles = 0.0
    for s, latency in ctx.vars.slot_latency.items():
        lat = float(latency.X)
        if lat <= 0:
            continue
        if compute_by_slot.get(s, 0.0) >= lat * 0.999:
            compute_cycles += lat
        else:
            transfer_cycles += lat
    per_iter = compute_cycles + transfer_cycles

    nodes = list(per_node.values())
    total_active = sum(d["latency_cycles"] for d in nodes) or 1
    weighted_util = sum((d["mac_spatial_utilization"] or 0.0) * d["latency_cycles"] for d in nodes) / total_active
    utils = [d["mac_spatial_utilization"] for d in nodes if d["mac_spatial_utilization"] is not None]
    offchip_id = space.accelerator.offchip_core_id
    degenerate_nodes = [name for name, d in per_node.items() if d["fallback"]]
    mac_utilization = guarded(
        "MAC utilization", end_to_end_mac_utilization, space.accelerator, total_mac_ops, total_latency
    )
    aggregate = {
        "compute_cores_available": sum(1 for c in space.accelerator.core_list if c.id != offchip_id),
        "compute_cores_used": len({c.id for n in space.ssc_nodes for c in lut.get_cores(n)}),
        "latency_weighted_mac_spatial_utilization": _json_scalar(weighted_util),
        "min_mac_spatial_utilization": _json_scalar(min(utils) if utils else None),
        "degenerate": bool(degenerate_nodes),
        "degenerate_nodes": degenerate_nodes,
    }
    return {
        "per_node": per_node,
        "bottleneck": {
            "compute_bound_cycles": int(compute_cycles),
            "transfer_bound_cycles": int(transfer_cycles),
            "compute_bound_pct": round(100.0 * compute_cycles / per_iter, 2) if per_iter else None,
            "transfer_bound_pct": round(100.0 * transfer_cycles / per_iter, 2) if per_iter else None,
        },
        "aggregate": aggregate | (mac_utilization or {}),
    }


def _json_scalar(v: Any) -> Any:
    """A solver or cost value as a JSON-safe scalar: an int when integral, else a float."""
    if v is None or isinstance(v, bool | str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return str(v)
    return int(f) if f.is_integer() else f


def overlap_section(ctx: FormulationContext, slack: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The inter-iteration overlap, the resources that bind it (those at the least ``slack``), each resource's
    slack, and the recurrence bound (RecMII; 0 for feed-forward) that separately caps it."""
    q, value = ctx.quantities, ctx.model.value
    slack = slack or []
    least = min((d["slack_cycles"] for d in slack), default=None)
    return {
        "overlap_cycles": int(value(q.get("overlap").expr)) if "overlap" in q else None,
        "binding_resources": [d["resource"] for d in slack if d["slack_cycles"] == least],
        "per_resource_slack": slack,
        "recurrence_bound_cycles": q.get("recurrence_bound").expr if "recurrence_bound" in q else 0,
    }


def resource_slack(ctx: FormulationContext) -> list[dict[str, Any]]:
    """Per resource its idle cycles within one iteration, least first. The overlap is at most the least slack
    across cores and links: a resource busy from an early to a late slot has none, and pins the overlap to zero."""
    q, value = ctx.quantities, ctx.model.value
    idle = q.indexed("idle_latency") if "idle_latency" in q else {}
    rows = [
        {
            "resource": str(res),
            "kind": "core" if isinstance(res, Core) else "link",
            "slack_cycles": int(round(value(quantity.expr))),
        }
        for res, quantity in idle.items()
    ]
    rows.sort(key=lambda d: d["slack_cycles"])
    return rows


def tensor_reuse_breakdown(ctx: FormulationContext, reuse_levels: dict[Tensor, int]) -> list[dict[str, Any]]:
    """Per tensor, largest first, the reuse the solve chose: for how many iterations it stays resident
    (``reuse_factor``; 1 is re-fetched every iteration), the level its reuse stops at, the tiles that takes and its
    loop nest, outermost first. A large tensor with ``reuse_factor`` 1 marks a bandwidth-bound fused schedule."""
    space = ctx.space
    rows: list[dict[str, Any]] = []
    for t, ssis in space.ssis.items():
        if not isinstance(t, Tensor):
            continue
        stop = reuse_levels.get(t)
        rows.append(
            {
                "tensor": t.name,
                "size_bits": int(t.size_bits()),
                "reuse_factor": space.reuse_levels.get((t, stop)) if stop is not None else None,
                "reuse_stop_level": stop,
                "on_chip_tiles": space.tiles_needed_levels.get((t, stop)) if stop is not None else None,
                "loop_nest_out_to_in": [repr(v) for v in reversed(ssis.variables)],
            }
        )
    rows.sort(key=lambda d: -d["size_bits"])
    return rows


def memory_occupancy(ctx: FormulationContext) -> list[dict[str, Any]]:
    """Per memory: the bits the solved placement keeps resident against its capacity, largest tensors first."""
    ledger = ctx.ledger
    cores = {c.id: c for c in ctx.space.accelerator.core_list}
    rows: list[dict[str, Any]] = []
    for core_id, terms in sorted(ledger.memory.items()):
        core = cores.get(core_id)
        if core is None:
            continue
        per_tensor: dict[str, int] = defaultdict(int)
        for indicator, bits, tensor_name in terms:
            if indicator.X > VAR_THRESHOLD:
                per_tensor[tensor_name] += bits
        per_tensor = {name: bits for name, bits in per_tensor.items() if bits > 0}
        if handed := ledger.handover_bits.get(core_id, 0):
            per_tensor["handover"] = handed
        resident = sum(per_tensor.values())
        capacity = int(ledger.bounds[(MEMORY_CAPACITY, core_id)] * 8)
        rows.append(
            {
                "core_id": core_id,
                "core_name": str(getattr(core, "type", "")) or str(core),
                "resident_bits": resident,
                "capacity_bits": capacity,
                "utilization": resident / capacity if capacity else None,
                "tensors": [
                    {"tensor": name, "bits": bits}
                    for name, bits in sorted(per_tensor.items(), key=lambda kv: -kv[1])[:_OCCUPANCY_TOP_TENSORS]
                ],
            }
        )
    return rows


def capacity_slack(ctx: FormulationContext, occupancy: list[dict[str, Any]] | None) -> dict[int, dict[str, float]]:
    """Unused capacity per core: memory in bytes, and each bounded limit (fifo depth, buffer descriptors) in slots."""
    ledger = ctx.ledger
    slack: dict[int, dict[str, float]] = {}
    for row in occupancy or ():
        slack.setdefault(row["core_id"], {})["memory_bytes"] = (row["capacity_bits"] - row["resident_bits"]) / 8
    for (kind, core_id), terms in ledger.loads.items():
        bound = ledger.bounds.get((kind, core_id))
        if bound is None:
            continue
        used = sum(count for var, count in terms if var.X > VAR_THRESHOLD)
        slack.setdefault(core_id, {})[kind.name] = bound - used
    return slack


def slot_latency_breakdown(
    ctx: FormulationContext, reuse: list[dict[str, Any]] | None, slack: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Per slot the compute and transfer contributors to its latency with the values its constraint is made of, and
    the totals, each resource's ``slack`` and each tensor's ``reuse``."""
    space, q, value = ctx.space, ctx.quantities, ctx.model.value
    breakdown: dict[int, dict[str, Any]] = {
        s: {"slot_latency_cycles": _json_scalar(latency.X), "compute_contributors": [], "transfer_contributors": []}
        for s, latency in ctx.vars.slot_latency.items()
    }
    for n in space.ssc_nodes:
        runtime = node_runtime(space, n)
        breakdown[space.slot_of[n]]["compute_contributors"].append(
            {
                "node": n.name,
                "cost_lut_core_count": len(space.cost_lut.get_cores(n)),
                "lut_latency_cycles": runtime,
                "active_fraction": _json_scalar(active_fraction(n, space.ssis)),
                "active_latency_cycles": get_active_latency(n, float(runtime), space.ssis),
            }
        )
    for (tr, choice), y in ctx.vars.y.items():
        if y.X < VAR_THRESHOLD:
            continue
        raw = int(space.transfer_latency_for_path(tr, choice))
        breakdown[space.slot_of[tr]]["transfer_contributors"].append(
            {
                "transfer": tr.name,
                "tensor_bits": int(tr.inputs[0].size_bits()) if tr.inputs else None,
                "min_link_bandwidth_bits_per_cycle": (
                    int(min(link.bandwidth for link in choice.links_used)) if choice.links_used else None
                ),
                "path_cycles": raw,
                "active_latency_cycles": int(get_active_latency(tr, float(raw), space.ssis)),
                "reuse_factor": _json_scalar(value(q.get("reuse_factor", tr).expr)),
                "latency_contribution_cycles": _json_scalar(value(q.get("transfer_latency", (tr, choice)).expr)),
            }
        )
    per_iteration = sum(latency.X for latency in ctx.vars.slot_latency.values())
    overlap = value(q.get("overlap").expr)
    shared = q.indexed("shared_busy") if "shared_busy" in q else {}
    return {
        "totals": {
            "iteration_latency_cycles": _json_scalar(per_iteration),
            "overlap_cycles": _json_scalar(overlap),
            "initiation_interval_cycles": _json_scalar(per_iteration - overlap),
            "total_latency_cycles": _json_scalar(value(q.get("total_latency").expr)),
            "shared_busy_cycles": {core: _json_scalar(value(busy.expr)) for core, busy in shared.items()},
        },
        "resource_slack": slack,
        "tensor_reuse": reuse,
        "slots": [{"slot": s, **breakdown[s]} for s in sorted(breakdown)],
    }


def solver_metrics(model: SolverModel, status: str) -> dict[str, Any]:
    """The solve's ``status`` and, for Gurobi, how much it searched, what it found, what that cost and how large the
    model was; each Gurobi value is None for another backend."""
    from stream.opt.solver import GurobiBackend  # noqa: PLC0415

    raw_model = model._model if isinstance(model, GurobiBackend) else None

    def _attr(name: str) -> Any | None:
        try:
            value = getattr(raw_model, name)
        except Exception:  # noqa: BLE001  # catches AttributeError and gp.GurobiError
            return None
        return None if isinstance(value, float) and not math.isfinite(value) else value

    return {
        "status": status,
        "search": {
            "nodes_explored": _attr("NodeCount"),
            "simplex_iterations": _attr("IterCount"),
            "barrier_iterations": _attr("BarIterCount"),
        },
        "solution": {
            "objective_value": _attr("ObjVal"),
            "objective_bound": _attr("ObjBound"),
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
    }
