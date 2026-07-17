"""Live DSE progress for the generic CO pipeline.

Emits an ordered, inspectable record of the framework's stages
(Parse -> Fuse -> Tile -> Allocate -> Schedule -> Cost) to ``progress.json`` as
the solve runs, so a just-launched mapping shows which framework steps are
running vs finished, each with a compact partial artifact drawn from the same
typed IR (WorkloadIR / AllocationIR) the finished run renders.

The mechanism is a pass-through :class:`ProgressProbeStage` interleaved into the
CO stage list. A probe fires an ``on_enter`` callback the instant control passes
through it -- i.e. the moment the stage above it finished its forward work -- so
progress advances without touching any real stage. Every callback is best-effort:
a failure to introspect the context is logged and swallowed so instrumentation can
never fail a solve.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)

# The inspection contract: the ordered DSE stages a mapping run passes through.
# (key, human label, one-line description of what the stage decides.)
PIPELINE_STAGES: list[tuple[str, str, str]] = [
    ("parse", "Parse", "Read the ONNX workload + accelerator into the graph IR"),
    ("fuse", "Fuse", "Split the graph into fusion groups at data-dependent cuts"),
    ("tile", "Tile", "Choose the steady-state tiling for each fused group"),
    ("core_cost", "Core cost", "Estimate each op's single-core latency & energy (ZigZag, per core)"),
    ("allocate", "Allocate", "Build the transfer/tensor graph, then assign each op to a core (CO solve)"),
    ("schedule", "Schedule", "Order the allocated ops into a steady-state schedule"),
    ("cost", "Cost", "Cost-model the schedule: latency, MAC utilization, bottleneck"),
]

_STATUS_RANK = {"pending": 0, "running": 1, "done": 2}
# The stages driven per fusion group by the inner pipeline (tile -> core cost -> allocate -> ...).
_INNER_KEYS = ("tile", "core_cost", "allocate", "schedule", "cost")
# Of those, the ones whose partial is sliced from the solved AllocationIR (core_cost comes from the
# cost LUT instead, captured at its own probe).
_ALLOC_FACET_KEYS = ("tile", "allocate", "schedule", "cost")


class ProgressTracker:
    """Accumulates DSE stage status + compact partials and writes progress.json atomically."""

    def __init__(self, path: str, operation: str):
        self.path = path
        self.operation = operation
        self.n_groups: int | None = None
        self.groups_completed = 0
        self.current_group: int | None = None
        self._stages: dict[str, dict[str, Any]] = {
            key: {"key": key, "label": label, "description": desc,
                  "status": "pending", "detail": None, "artifact": None}
            for key, label, desc in PIPELINE_STAGES
        }
        self._order = [key for key, _, _ in PIPELINE_STAGES]
        self._write()

    # -- status transitions (monotonic: pending -> running -> done) --------

    def _advance(self, key: str, status: str) -> None:
        st = self._stages[key]
        if _STATUS_RANK[status] >= _STATUS_RANK[st["status"]]:
            st["status"] = status

    def mark(self, key: str, status: str = "running", *, detail: str | None = None,
             artifact: Any = None) -> None:
        self._advance(key, status)
        if detail is not None:
            self._stages[key]["detail"] = detail
        if artifact is not None:
            self._stages[key]["artifact"] = artifact
        self._write()

    def set_groups(self, n: int) -> None:
        self.n_groups = n
        self._write()

    def _group_detail(self, index: int) -> str:
        return f"group {index + 1}/{self.n_groups}" if self.n_groups else f"group {index + 1}"

    def reach_inner(self, key: str, index: int) -> None:
        """Fusion group ``index`` has reached inner stage ``key`` — it is now running. Marked
        progressively (tile, then allocate, …) so a stopped run shows exactly how far it got."""
        self.current_group = index
        self._advance(key, "running")
        self._stages[key]["detail"] = self._group_detail(index)
        self._write()

    def costed_group(self, index: int, cost_lut: Any) -> None:
        """Single-core cost estimation done for group ``index``: attach the per-node ZigZag latency &
        energy, then mark that allocation has begun."""
        art = self._stages["core_cost"]["artifact"] or {"groups": []}
        art["groups"].append({"group": index, "nodes": core_cost_artifact(cost_lut)})
        self._stages["core_cost"]["artifact"] = art
        self._advance("core_cost", "running")
        self.reach_inner("allocate", index)

    def solved_group(self, index: int, alloc: dict[str, Any] | None) -> None:
        """A fusion group solved: attach its per-stage partial IR facets (from the AllocationIR), mark
        allocate/schedule/cost reached, then record the group done (which flips the inner stages to
        done once all groups pass)."""
        facets = inner_facets(alloc)
        for key in _ALLOC_FACET_KEYS:
            art = self._stages[key]["artifact"] or {"groups": []}
            art["groups"].append({"group": index, **(facets.get(key) or {})})
            self._stages[key]["artifact"] = art
        for key in ("allocate", "schedule", "cost"):
            self._advance(key, "running")
            self._stages[key]["detail"] = self._group_detail(index)
        self.group_done(index)

    def group_done(self, index: int) -> None:
        """Record that fusion group ``index`` finished its inner pipeline."""
        self.groups_completed = max(self.groups_completed, index + 1)
        self.current_group = index
        # An inner stage is 'done' only once every group has passed through it.
        if self.n_groups is not None and self.groups_completed >= self.n_groups:
            for key in _INNER_KEYS:
                self._advance(key, "done")
        self._write()

    def fail(self, reason: str | None) -> None:
        """The solve stopped (e.g. an infeasible mapping). Mark the stage it stopped in as failed, the
        stages it had already cleared as done, and leave the ones it never reached pending."""
        first_line = (reason or "").strip().splitlines()
        msg = first_line[0][:160] if first_line else "stopped"
        running = [k for k in _INNER_KEYS if self._stages[k]["status"] == "running"]
        if running:
            for key in running[:-1]:
                self._stages[key]["status"] = "done"  # earlier reached stages cleared for this group
            failed = running[-1]
            self._stages[failed]["status"] = "failed"
            self._stages[failed]["detail"] = msg
        else:
            # Stopped in a top-level stage (parse/fuse): fail the first not-yet-done stage.
            for key in self._order:
                if self._stages[key]["status"] != "done":
                    self._stages[key]["status"] = "failed"
                    self._stages[key]["detail"] = msg
                    break
        self._write()

    def finish(self) -> None:
        """Mark every stage done (called once the solve returns)."""
        for key in self._order:
            self._advance(key, "done")
        self._write()

    def _write(self) -> None:
        payload = {
            "schema_version": "1.0",
            "operation": self.operation,
            "n_groups": self.n_groups,
            "groups_completed": self.groups_completed,
            "current_group": self.current_group,
            "stages": [self._stages[k] for k in self._order],
        }
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.path)  # atomic: pollers never see a half-written file
        except Exception as exc:  # noqa: BLE001 -- progress IO must never fail a solve
            logger.warning(f"[progress] failed to write {self.path}: {exc}")


class ProgressProbeStage(Stage):
    """A no-op pass-through stage that fires ``on_enter`` when control reaches it.

    Reaching a probe means the preceding stage completed its forward work, so the
    probe is the boundary marker for 'stage above me finished'. It yields exactly
    what its sub-pipeline yields (one context per group), preserving the inner
    pipeline's single-result contract.
    """

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext,
                 *, on_enter: Callable[[StageContext], None] | None = None):
        super().__init__(list_of_callables, ctx)
        self._on_enter = on_enter

    def is_leaf(self) -> bool:
        return False

    def run(self):
        if self._on_enter is not None:
            try:
                self._on_enter(self.ctx)
            except Exception as exc:  # noqa: BLE001 -- a probe must never fail the solve
                logger.warning(f"[progress] probe on_enter failed: {exc}")
        sub = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub.run()


# --------------------------------------------------------------------------- #
# Compact, directly-renderable partial artifacts drawn from the typed IR.      #
# Every builder is defensive: any failure returns None (partial simply absent).#
# --------------------------------------------------------------------------- #

def _node_op(node: Any) -> str:
    for attr in ("type", "op_type", "op"):
        val = getattr(node, attr, None)
        if val:
            return str(val)
    return node.__class__.__name__


def parse_artifact(ctx: StageContext) -> dict[str, Any] | None:
    """Compact summary of the parsed workload + accelerator (stage 1)."""
    try:
        workload = ctx.get("workload")
        nodes = list(workload.get_computation_nodes()) if workload is not None else []
        op_hist = Counter(_node_op(n) for n in nodes)
        accel = ctx.get("accelerator")
        accel_name = getattr(accel, "name", None) or (os.path.basename(str(accel)) if accel else None)
        cores = None
        try:
            cores = len(list(accel.core_list))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            cores = None
        return {
            "n_nodes": len(nodes),
            "op_types": dict(sorted(op_hist.items(), key=lambda kv: -kv[1])),
            "accelerator": accel_name,
            "cores": cores,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[progress] parse_artifact failed: {exc}")
        return None


def fuse_artifact(ctx: StageContext) -> tuple[dict[str, Any] | None, int | None]:
    """Compact summary of the fusion groups (stage 2). Returns (artifact, n_groups).

    Uses ``sub_workloads`` when the pipeline splits into fusion groups (generic / fixed multi-group);
    falls back to the single parsed ``workload`` as one group for a single-shot fixed mapping."""
    try:
        sub_workloads = ctx.get("sub_workloads")
        if sub_workloads:
            groups = []
            for i, wl in enumerate(sub_workloads):
                nodes = list(wl.get_computation_nodes())
                ops = Counter(_node_op(n) for n in nodes)
                groups.append({
                    "name": f"group_{i}",
                    "n_nodes": len(nodes),
                    "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
                })
            return {"n_groups": len(groups), "groups": groups}, len(groups)
        # Single-shot fixed mapping: no fusion split — the whole workload is solved as one group.
        workload = ctx.get("workload")
        if workload is not None:
            nodes = list(workload.get_computation_nodes())
            ops = Counter(_node_op(n) for n in nodes)
            return (
                {"n_groups": 1, "groups": [{"name": "group_0", "n_nodes": len(nodes),
                                            "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1]))}]},
                1,
            )
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[progress] fuse_artifact failed: {exc}")
        return None, None


def _tile_facet(alloc: dict[str, Any]) -> dict[str, Any]:
    tiling = alloc.get("tiling") or {}
    return {
        "fusion_splits": tiling.get("fusion_splits", []),
        "intra_core": tiling.get("intra_core", {}),
        "inter_core": tiling.get("inter_core", {}),
    }


def _allocate_facet(alloc: dict[str, Any]) -> dict[str, Any]:
    nodes = alloc.get("mapping_nodes") or {}
    placement = {}
    for name, node in nodes.items():
        res = node.get("resource_allocation") or []
        cores = sorted({r.get("id") for slot in res for r in slot
                        if isinstance(r, dict) and r.get("type") == "core" and r.get("id") is not None})
        placement[name] = cores
    # The TETRA transfer/tensor-aware graph the CO solve is built on (compute + injected transfer
    # nodes). Already in the IR as steady_state.tiled_graph; surface it so Allocate can visualize it.
    graph = ((alloc.get("steady_state") or {}) or {}).get("tiled_graph") if alloc.get("steady_state") else None
    return {"placement": placement, "graph": graph}


def _schedule_facet(alloc: dict[str, Any]) -> dict[str, Any]:
    return {"latency": alloc.get("latency")}


def _cost_facet(alloc: dict[str, Any]) -> dict[str, Any]:
    perf = alloc.get("performance") or {}
    return {
        "latency": alloc.get("latency"),
        "bottleneck": perf.get("bottleneck"),
        "aggregate": perf.get("aggregate"),
    }


def inner_facets(alloc: dict[str, Any] | None) -> dict[str, Any]:
    """Slice one group's AllocationIR dict into the tile/allocate/schedule/cost facets."""
    if not alloc:
        return {}
    facets: dict[str, Any] = {}
    for key, fn in (("tile", _tile_facet), ("allocate", _allocate_facet),
                    ("schedule", _schedule_facet), ("cost", _cost_facet)):
        try:
            facets[key] = fn(alloc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[progress] {key} facet failed: {exc}")
    return facets


def core_cost_artifact(cost_lut: Any) -> list[dict[str, Any]]:
    """Per-node single-core cost from the CoreCostLUT: the ZigZag latency & energy of each op on the
    best (min-latency) core it can run on. Best-effort — any failure yields an empty list."""
    rows: list[dict[str, Any]] = []
    if cost_lut is None:
        return rows
    try:
        for node in cost_lut.get_nodes():
            best = None
            for core in cost_lut.get_cores(node):
                try:
                    entry = cost_lut.get_cost(node, core)
                except Exception:  # noqa: BLE001
                    continue
                lat = getattr(entry, "latency_total", None)
                if best is None or (lat is not None and lat < best[1]):
                    best = (core, lat if lat is not None else float("inf"), entry)
            if best is None:
                continue
            core, _lat, entry = best
            ideal = getattr(entry, "ideal_cycle", None)
            lat = getattr(entry, "latency_total", None)
            rows.append({
                "node": getattr(node, "name", str(node)),
                "op": getattr(node, "type", None),
                "core": getattr(core, "id", None),
                "latency": lat,
                "energy": getattr(entry, "energy_total", None),
                "ideal_cycle": ideal,
                # Compute efficiency: how close the single-core estimate runs to the compute-ideal floor.
                "efficiency": (ideal / lat) if (ideal and lat) else None,
            })
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[progress] core_cost_artifact failed: {exc}")
    return rows


def alloc_partial(scheduler: Any) -> dict[str, Any] | None:
    """Build the AllocationIR dict for a solved scheduler (the per-group partial). Best-effort: any
    failure returns None so a caller that hand-drives the pipeline never fails on instrumentation."""
    if scheduler is None:
        return None
    try:
        from stream.ir.allocation import AllocationIR  # noqa: PLC0415 -- avoid import cycle at module load

        return AllocationIR.from_internal(scheduler).model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[progress] alloc_partial failed: {exc}")
        return None


def group_index_from_ctx(ctx: StageContext) -> int:
    """Recover the fusion-group index from the per-group ``output_path`` (``.../group_{i}``)."""
    match = re.search(r"group_(\d+)", os.path.basename(str(ctx.get("output_path") or "")))
    return int(match.group(1)) if match else 0


# Stage class names, keyed off ``__name__`` so we never import the stage classes here (avoids cycles).
_PARSE_ONNX = "ONNXModelParserStage"
_PARSE_ACCEL = "AcceleratorParserStage"
_FUSE_PRODUCERS = ("GenericMappingGenerationStage", "FixedMappingGenerationStage")
_MAPPING_PARSER = "MappingParserStage"
_TILING = "TilingGenerationStage"
_CORE_COST = "CoreCostEstimationStage"
_CONSTRAINT_OPT = "ConstraintOptimizationAllocationStage"


def _stage_name(stage: StageCallable) -> str:
    return getattr(stage, "__name__", type(stage).__name__)


def instrument_stages(stages: list[StageCallable], tracker: ProgressTracker) -> list[StageCallable]:
    """Interleave inert progress probes into a CO stage list so it emits live progress via ``tracker``.

    Placement is driven by stage class name, so one function covers every CO shape: the generic
    multi-group pipeline, a single-shot fixed mapping, and the fixed multi-group codegen pipeline. For
    the generic shape the placement is identical to hand-writing the probes. Probes are pass-through
    wrappers that never change which real stages run.
    """
    names = [_stage_name(s) for s in stages]
    has_onnx = _PARSE_ONNX in names
    has_fuse_producer = any(n in _FUSE_PRODUCERS for n in names)

    def _cb_parse(ctx: StageContext) -> None:
        art = parse_artifact(ctx)
        detail = f"{art['n_nodes']} nodes" if art and art.get("n_nodes") is not None else None
        tracker.mark("parse", "done", detail=detail, artifact=art)
        tracker.mark("fuse", "running")

    def _cb_fuse(ctx: StageContext) -> None:
        art, n_groups = fuse_artifact(ctx)
        if n_groups is not None:
            tracker.set_groups(n_groups)
        detail = f"{n_groups} fusion group{'s' if n_groups != 1 else ''}" if n_groups is not None else None
        tracker.mark("fuse", "done", detail=detail, artifact=art)

    def _cb_begin(ctx: StageContext) -> None:
        tracker.reach_inner("tile", group_index_from_ctx(ctx))

    def _cb_tiled(ctx: StageContext) -> None:
        tracker.reach_inner("core_cost", group_index_from_ctx(ctx))

    def _cb_costed(ctx: StageContext) -> None:
        tracker.costed_group(group_index_from_ctx(ctx), ctx.get("cost_lut"))

    def _cb_solved(ctx: StageContext) -> None:
        tracker.solved_group(group_index_from_ctx(ctx), alloc_partial(ctx.get("scheduler")))

    def _probe(cb: Callable[[StageContext], None]) -> StageCallable:
        import functools  # noqa: PLC0415

        return functools.partial(ProgressProbeStage, on_enter=cb)

    out: list[StageCallable] = []
    parse_emitted = False
    fuse_emitted = False
    for i, stage in enumerate(stages):
        name = names[i]
        if name == _TILING:
            out.append(_probe(_cb_begin))  # tile begins
        out.append(stage)
        if not parse_emitted and (
            (has_onnx and name == _PARSE_ONNX) or (not has_onnx and name == _PARSE_ACCEL)
        ):
            out.append(_probe(_cb_parse))  # parse done
            parse_emitted = True
        if not fuse_emitted and (
            (has_fuse_producer and name in _FUSE_PRODUCERS)
            or (not has_fuse_producer and name == _MAPPING_PARSER)
        ):
            out.append(_probe(_cb_fuse))  # fusion groups decided
            fuse_emitted = True
        if name == _TILING:
            out.append(_probe(_cb_tiled))  # tiled -> core-cost estimation begins
        if name == _CORE_COST:
            out.append(_probe(_cb_costed))  # single-core cost done -> allocation begins
        if name == _CONSTRAINT_OPT:
            out.append(_probe(_cb_solved))  # solved -> per-stage partial IR facets
    return out
