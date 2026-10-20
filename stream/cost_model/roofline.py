"""The roofline of an accelerator, and where each solved fusion group sits on it.

The roofline is the hardware's: a compute roof per class of compute core, and a bandwidth ceiling per level of its
memory hierarchy (off-chip memory, the links to it, on-chip memory, the links between cores). A point is one solved
group, whose operations per cycle (MACs, or elementwise operations against the vector roof for a group without MACs)
are read against the bytes it moved through each level, so one point has an operational intensity per level.
"""

from __future__ import annotations

from collections import defaultdict
from math import prod
from typing import TYPE_CHECKING, Any

from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.families.traffic import link_cores, link_name, port_name
from stream.opt.allocation.constraint_optimization.report import compute_units, runs_macs

if TYPE_CHECKING:
    from stream.allocation.allocation import Allocation
    from stream.hardware.architecture.accelerator import Accelerator

OFFCHIP_MEMORY = "off-chip memory"
OFFCHIP_LINKS = "off-chip links"
ONCHIP_MEMORY = "on-chip memory"
CORE_LINKS = "core-to-core links"
LEVELS = (OFFCHIP_MEMORY, OFFCHIP_LINKS, ONCHIP_MEMORY, CORE_LINKS)
COMPUTE = "compute"
ARRAY_FILL = "array fill"

SATURATED = 0.95
"""A resource this busy is reported as saturated."""
GAP = 0.5
"""A point below this fraction of its lowest ceiling is reported as below it."""


def hardware_roofline(accelerator: Accelerator) -> dict[str, Any]:
    """The roofs and ceilings of ``accelerator``, each with the resources it sums."""
    offchip = accelerator.offchip_core_id
    roofs: dict[tuple[str, Any], dict[str, Any]] = {}
    for core in accelerator.core_list:
        for name, served, units in _core_roofs(accelerator, core):
            roof = roofs.setdefault((name, served), {"name": name, "ops_per_cycle": 0, "cores": []})
            roof["ops_per_cycle"] += units
            roof["cores"].append(core.id)
    for (_, served), roof in roofs.items():
        if served:
            roof["name"] += "@" + ",".join(map(str, sorted(roof["cores"])))
    levels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for port in accelerator.ports:
        levels[OFFCHIP_MEMORY if offchip in port.core_ids else ONCHIP_MEMORY].append(
            {
                "resource": _port_name(port_name(port.key), port.core_ids),
                "cores": list(port.core_ids),
                "bytes_per_cycle": port.bits_per_cycle / 8,
            }
        )
    for core_id, model in accelerator.bandwidth.items():
        levels[OFFCHIP_MEMORY if core_id == offchip else ONCHIP_MEMORY].append(
            {"resource": _port_name("measured", (core_id,)), "cores": [core_id], "bytes_per_cycle": model.ceiling / 8}
        )
    for link in accelerator.links:
        cores = sorted(link_cores(link))
        levels[OFFCHIP_LINKS if offchip in cores else CORE_LINKS].append(
            {"resource": link_name(link), "cores": cores, "bytes_per_cycle": link.bandwidth / 8}
        )
    return {
        "roofs": sorted(roofs.values(), key=lambda roof: -roof["ops_per_cycle"]),
        "levels": [
            {
                "name": name,
                "bytes_per_cycle": sum(r["bytes_per_cycle"] for r in levels[name]),
                "resources": levels[name],
            }
            for name in LEVELS
            if levels[name]
        ],
    }


def group_point(allocation: Allocation, hardware: dict[str, Any]) -> dict[str, Any]:
    """Where the solved group ``allocation`` sits on ``hardware``'s roofline, at the latency its solve models, with
    what holds it down."""
    performance = allocation.solution.performance or {}
    aggregate = performance.get("aggregate") or {}
    latency = allocation.solution.latency.total
    macs = float(aggregate.get("total_mac_ops") or 0)
    workload = allocation.problem.source_workload
    nodes = workload.get_computation_nodes()
    operations = macs or float(sum(prod(workload.get_dimension_size(d) for d in workload.get_dims(cn)) for cn in nodes))
    roof = _roof(hardware, allocation, macs)
    traffic = _traffic(performance.get("memory_ports") or [], hardware, allocation.problem.iterations)
    peak = roof["ops_per_cycle"] if roof else 0
    achieved = operations / latency if latency else 0.0
    levels = []
    for level in hardware["levels"]:
        moved, busiest = traffic.get(level["name"], (0.0, None))
        intensity = operations / moved if moved else None
        attainable = level["bytes_per_cycle"] * intensity if intensity is not None else None
        levels.append(
            {
                "name": level["name"],
                "bytes": moved,
                "intensity": intensity,
                "attainable": attainable,
                "busiest": busiest,
            }
        )
    ceilings = [(COMPUTE, float(peak))] + [(lv["name"], lv["attainable"]) for lv in levels if lv["attainable"]]
    bound, attainable = min(ceilings, key=lambda c: c[1]) if peak else (None, None)
    # Zero when no node reports its spatial utilization (an AIE kernel): unknown, so no ceiling.
    spatial = aggregate.get("latency_weighted_mac_spatial_utilization") or None
    if spatial is not None and peak:
        ceilings.append((ARRAY_FILL, float(peak) * spatial))
    limiter, limit = min(ceilings, key=lambda c: c[1]) if peak else (None, None)
    point = {
        "layers": [cn.name for cn in nodes],
        "operators": sorted({cn.type for cn in nodes}),
        "roof": roof["name"] if roof else None,
        "macs": macs,
        "operations": operations,
        "latency_cycles": latency,
        "fill_cycles": int(allocation.solution.latency.fill),
        "iterations": allocation.problem.iterations,
        "ops_per_cycle": achieved,
        "levels": levels,
        "bound": bound,
        "attainable": attainable,
        "efficiency": achieved / attainable if attainable else None,
        "spatial_utilization": spatial,
        "limiter": limiter,
        "limit": limit,
    }
    return point | {"diagnosis": _diagnosis(point), "signature": _signature(allocation, latency, levels)}


def run_roofline(hardware: dict[str, Any], points: dict[int, dict[str, Any]], n_groups: int) -> dict[str, Any]:
    """The run's roofline: ``hardware`` and the ``points`` of its solved groups by index, groups with the same
    operators, shapes, placement and cost merged into one point sized by its share of the run's latency."""
    merged: dict[str, dict[str, Any]] = {}
    for index in sorted(points):
        point = points[index]
        if point["signature"] in merged:
            merged[point["signature"]]["groups"].append(index)
        else:
            merged[point["signature"]] = {k: v for k, v in point.items() if k != "signature"} | {"groups": [index]}
    total = sum(p["latency_cycles"] for p in points.values())
    out = []
    for point in merged.values():
        n = len(point["groups"])
        out.append(point | {"multiplicity": n, "share": n * point["latency_cycles"] / total if total else 0.0})
    bound_share: dict[str, float] = defaultdict(float)
    limiter_share: dict[str, float] = defaultdict(float)
    for point in out:
        bound_share[point["bound"] or "unplotted"] += point["share"]
        limiter_share[point["limiter"] or "unplotted"] += point["share"]
    return {
        "hardware": hardware,
        "points": out,
        "groups_solved": len(points),
        "n_groups": n_groups,
        "summary": {
            "latency_cycles": total,
            "macs": sum(p["macs"] * p["multiplicity"] for p in out),
            "operations": sum(p["operations"] * p["multiplicity"] for p in out),
            "bound_share": dict(bound_share),
            "limiter_share": dict(limiter_share),
        },
    }


def _diagnosis(point: dict[str, Any]) -> list[dict[str, Any]]:
    """What holds the point down, the lowest ceiling first: the roofline's bound or the fill of the MAC array, a level
    it runs above (whose rate the solve did not enforce), a resource it saturates, and its gap below the ceiling."""
    limiter = point["limiter"]
    if limiter is None:
        return [{"code": "no_roof"}]
    if limiter == COMPUTE:
        found = [{"code": "compute_bound"}]
    elif limiter == ARRAY_FILL:
        found = [{"code": "array_underfilled", "spatial_utilization": point["spatial_utilization"]}]
    else:
        found = [{"code": "memory_bound", "level": limiter}]
    for level in point["levels"]:
        busiest = level["busiest"]
        if level["attainable"] and point["ops_per_cycle"] > level["attainable"] * (1 + 1e-9):
            found.append({"code": "exceeds_ceiling", "level": level["name"], **(busiest or {})})
        elif busiest and (busiest["utilization"] or 0) >= SATURATED and level["name"] != limiter:
            found.append({"code": "resource_saturated", "level": level["name"], **busiest})
    if point["limit"] and point["ops_per_cycle"] / point["limit"] < GAP:
        latency = point["latency_cycles"]
        found.append(
            {
                "code": "below_ceiling",
                "fraction": point["ops_per_cycle"] / point["limit"],
                "fill_share": point["fill_cycles"] / latency if latency else None,
            }
        )
    return found


def _placed_cores(allocation: Allocation) -> dict[str, set[int]]:
    """The cores each computation node runs on, by node name."""
    return {
        cn.name: {
            core.id
            for slot in allocation.mapping.get(cn).resource_allocation or ()
            for core in slot
            if isinstance(core, Core)
        }
        for cn in allocation.problem.workload.get_computation_nodes()
        if cn in allocation.mapping
    }


def _roof(hardware: dict[str, Any], allocation: Allocation, macs: float) -> dict[str, Any] | None:
    """The roof a group is read against: the MAC roof for MAC work, else the vector roofs of the cores it runs on when
    each has one, else the MAC roof of the generic cores running it."""
    cores = set().union(*_placed_cores(allocation).values())
    vector = [r for r in hardware["roofs"] if r["name"].startswith("vector") and cores & set(r["cores"])]
    covered = {core for roof in vector for core in roof["cores"]}
    if not macs and cores and cores <= covered:
        ops = sum(roof["ops_per_cycle"] for roof in vector)
        return {"name": " + ".join(roof["name"] for roof in vector), "ops_per_cycle": ops, "cores": sorted(covered)}
    return next((roof for roof in hardware["roofs"] if roof["name"] == "MAC"), None)


def _traffic(rows: list[dict[str, Any]], hardware: dict[str, Any], iterations: int) -> dict[str, tuple]:
    """Per level, the bytes the run moved through its resources and the busiest of them."""
    level_of = {r["resource"]: level["name"] for level in hardware["levels"] for r in level["resources"]}
    moved: dict[str, float] = defaultdict(float)
    busiest: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["resource"] if row["kind"] == "link" else _port_name(row["resource"], row["core_ids"])
        if (level := level_of.get(name)) is None:
            continue
        moved[level] += row["bits_per_iteration"] * iterations / 8
        if level not in busiest or (row["utilization"] or 0) > (busiest[level]["utilization"] or 0):
            busiest[level] = {"resource": name, "utilization": row["utilization"]}
    return {level: (moved[level], busiest.get(level)) for level in moved}


def _signature(allocation: Allocation, latency: int, levels: list[dict[str, Any]]) -> str:
    """What makes two groups the same point: their operators and shapes in order, where each runs, and their cost."""
    workload = allocation.problem.source_workload
    placed = _placed_cores(allocation)
    nodes = [
        (cn.type, [workload.get_dimension_size(d) for d in workload.get_dims(cn)], sorted(placed.get(cn.name, ())))
        for cn in workload.get_computation_nodes()
    ]
    return repr((nodes, latency, [round(level["bytes"]) for level in levels]))


def _core_roofs(accelerator: Accelerator, core: Core) -> list[tuple[str, Any, float]]:
    """The roofs ``core`` adds operations per cycle to, each with the operators its cores serve when that splits it:
    its operational array to the MAC roof, or to the vector roof of the operators it serves; without one, its kernel
    library's MAC tile and vector rate to the MAC and vector roofs."""
    if units := compute_units(accelerator, core):
        return [("MAC", None, units) if runs_macs(core) else ("vector", frozenset(core.operator_types or ()), units)]
    library = accelerator.kernel_library
    if library is None or core.type != "compute":
        return []
    families = library.families
    mac = families.get("matmul")
    roofs = [("MAC", None, prod(mac.mac.values()))] if mac is not None and mac.mac else []
    return roofs + ([("vector", None, families["vector"].ops_per_cycle)] if "vector" in families else [])


def _port_name(port: str, core_ids: Any) -> str:
    return f"{port}@{','.join(map(str, sorted(core_ids)))}"
