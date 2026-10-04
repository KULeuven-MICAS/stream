import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import replace
from math import ceil
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import yaml

# GRB supplies the callback codes of _mip_progress_callback and the Gurobi status names of the solve
# summary; gurobipy is optional, so GRB is None without it and only the Gurobi solve path touches it.
try:
    from gurobipy import GRB
except ModuleNotFoundError:
    GRB = None  # type: ignore[assignment]

from stream.allocation.problem import SteadyStateProblem
from stream.allocation.solution import AllocationSolution, Latency, end_to_end_mac_utilization
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.ir.infeasibility import (
    ConstraintTermIR,
    ImplicatedResourceIR,
    InfeasibilityReportIR,
    InfeasibleAllocationError,
    ResourceRefIR,
    StructuralConflictIR,
    TileDimIR,
    UnmetConstraintIR,
)
from stream.mapping.mapping import Resource
from stream.opt.allocation.constraint_optimization.families import (
    FamilySelection,
    ObjectiveFamily,
    ReportingFamily,
)
from stream.opt.allocation.constraint_optimization.families.memory import capacity_screen
from stream.opt.allocation.constraint_optimization.formulation import DecisionVariables, FormulationContext
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace, Placement
from stream.opt.allocation.constraint_optimization.timeslot_allocation import _resource_key
from stream.opt.allocation.constraint_optimization.utils import active_fraction, get_active_latency
from stream.opt.solver import (
    ObjectiveLevel,
    SolverBackend,
    SolverModel,
    SolverParams,
    SolverVar,
    SolverVarType,
    create_solver,
)
from stream.profiling import span
from stream.workload.steady_state.iteration_space import Reuse
from stream.workload.steady_state.node import Node
from stream.workload.workload import Tensor, TransferNode

_logger = logging.getLogger(__name__)

_OCCUPANCY_TOP_TENSORS = 8

TensorReuseLevels: TypeAlias = dict[Tensor, int]
TensorDepths: TypeAlias = dict[Tensor, int]
TensorAlloc: TypeAlias = dict[Tensor, Placement]
TransferAlloc: TypeAlias = dict[TransferNode, MulticastPathPlan]
MemoryAlloc: TypeAlias = dict[TransferNode, Placement]


class TransferAndTensorAllocator:
    """The allocation model of a steady-state problem: the core decision variables -- where every movable tensor
    lives, which route each transfer takes, where each tensor's reuse stops -- the constraints and objective levels
    its families build on them, and the allocation its solve reads back."""

    VAR_THRESHOLD = 0.5

    def __init__(
        self,
        problem: SteadyStateProblem,
        *,
        families: FamilySelection,
        output_path: str = "",
        backend: str = "ORTOOLS_GSCIP",
    ):
        self.families = families
        self.output_path = output_path
        self.space = DecisionSpace(problem)
        self.model: SolverModel = create_solver(SolverBackend[backend], "transfer_tensor_alloc")
        self.model.set_param(SolverParams.VERBOSITY, 1)
        self.model.set_param(SolverParams.LOG_TO_CONSOLE, 0)
        self.quantities = QuantityRegistry()
        self.optimization_trace: list[dict[str, float | str | None]] = []
        self._build_model()

    def _build_model(self) -> None:
        with span("capacity_screen"):
            capacity_screen(self.space, self.model)
        with span("variables"):
            self.vars = self._create_variables()
        self.context = FormulationContext(self.space, self.vars, self.model, self.quantities)
        for name, build in self.families.steps:
            with span(f"family_{name}"):
                build(self.context, self.quantities)
        with span("objective"):
            self.objective = self._objective_levels()
            self.model.set_lexicographic_objectives(list(self.objective.values()), sense="minimize")

    def _create_variables(self) -> DecisionVariables:
        space, model = self.space, self.model
        x: dict[tuple[Tensor, Placement], SolverVar] = {}
        for t in space.tensor_var:
            for choice in space.tensor_choices[t]:
                choice_name = "__".join(_resource_key(c) for c in choice)
                x[(t, choice)] = model.add_var(vtype=SolverVarType.BINARY, name=f"x_{t.name}_{choice_name}")
        y: dict[tuple[TransferNode, MulticastPathPlan], SolverVar] = {}
        for tr in space.transfer_nodes:
            for i, choice in enumerate(space.path_choices[tr]):
                y[(tr, choice)] = model.add_var(vtype=SolverVarType.BINARY, name=f"y_{tr.name}_choice_{i}")
        z_stop = self._create_reuse_vars()
        z_single: dict[tuple[Tensor, int], SolverVar] = {}
        for t in space.tensors_to_optimize_reuse_for:
            for stop in range(len(space.ssis[t].get_applicable_temporal_variables())):
                if space.may_single_buffer(t, stop):
                    v = model.add_var(vtype=SolverVarType.BINARY, name=f"zSingle_{t.name}_L{stop}")
                    model.add_constr(v <= z_stop[(t, stop)], name=f"zSingle_AtStop_{t.name}_L{stop}")
                    z_single[(t, stop)] = v
        slot_latency: dict[int, SolverVar] = {}
        for s in range(space.max_slot + 1):
            slot_latency[s] = model.add_var(vtype=SolverVarType.INTEGER, name=f"L_{s}")
            self.quantities.add("slot_latency", slot_latency[s]._raw, index=s)
        return DecisionVariables(x=x, y=y, z_stop=z_stop, z_single=z_single, slot_latency=slot_latency)

    def _create_reuse_vars(self) -> dict[tuple[Tensor, int], SolverVar]:
        """One binary per tensor and reuse stop, exactly one of which is set, at or beyond a declared reuse."""
        space, model = self.space, self.model
        z_stop: dict[tuple[Tensor, int], SolverVar] = {}
        optimized = set(space.tensors_to_optimize_reuse_for)
        for t in space.workload.tensors:
            levels = len(space.ssis[t].get_applicable_temporal_sizes())
            for stop in range(-1, levels):
                z_stop[(t, stop)] = model.add_var(vtype=SolverVarType.BINARY, name=f"zStop_{t.name}_L{stop}")
            model.add_constr(
                model.quicksum(z_stop[(t, s)]._raw for s in range(-1, levels)) == 1,
                name=f"zStop_Choose_One_{t.name}",
            )
            if t in optimized:
                continue
            reuses = space.ssis[t].get_temporal_reuses()
            stop = next((i for i in range(len(reuses) - 1, -1, -1) if reuses[i] == Reuse.REUSE), -2)
            assert stop >= -1, f"Something went wrong for {t.name} REUSE indexing: {reuses}"
            # The declared reuse is a floor, not a target: holding a tensor across more
            # levels than asked for only removes transfers, and the capacity, routing and
            # compute-compatibility constraints already say when that does not fit.
            model.add_constr(
                model.quicksum(z_stop[(t, s)]._raw for s in range(stop, levels)) == 1,
                name=f"zStop_AtLeast_{t.name}_L{stop}",
            )
        return z_stop

    def _objective_levels(self) -> dict[str, ObjectiveLevel]:
        """The families' objective levels by name, highest priority first, those of one name summed into one."""
        levels: dict[str, ObjectiveLevel] = {}
        for family in self.families.families:
            if not isinstance(family, ObjectiveFamily):
                continue
            for level in family.objective(self.context, self.quantities):
                if (merged := levels.get(level.name)) is None:
                    levels[level.name] = level
                elif merged.priority != level.priority:
                    raise ValueError(
                        f"Objective level {level.name!r} has priority {level.priority} in {family.name!r}, "
                        f"{merged.priority} elsewhere"
                    )
                else:
                    levels[level.name] = replace(merged, expr=merged.expr + level.expr)
        if "total_latency" not in self.quantities:
            raise ValueError("The allocation needs the overlap family, which defines the latency objective")
        return dict(sorted(levels.items(), key=lambda item: -item[1].priority))

    # ------------------------------------------------------------------ #
    # infeasibility diagnosis                                            #
    # ------------------------------------------------------------------ #
    # Constraint family -> human reason. New resource-bound families add one line here.
    _KIND_REASON: dict[str, str] = {
        "memory_capacity": "on-chip memory capacity exceeded",
        "link_contention": "communication link over-subscribed",
        "object_fifo_depth": "object-FIFO depth exceeded",
        "buffer_descriptors": "buffer-descriptor count exceeded",
        "dma_channels": "DMA channel limit exceeded",
    }
    # Untagged IIS constraint name-prefix -> resource-limit family. Only these contribute a *reason*;
    # placement/indicator constraints (u_, z_, one-hot selectors) merely attribute to the resource.
    _LIMIT_PREFIXES: tuple[tuple[str, str], ...] = (
        ("mem_cap", "memory_capacity"),
        ("memload", "memory_capacity"),
        ("aie2_obj_fifo", "object_fifo_depth"),
        ("objfifo", "object_fifo_depth"),
        ("aie2_bd", "buffer_descriptors"),
        ("bddepth", "buffer_descriptors"),
        ("link_used", "link_contention"),
        ("dma", "dma_channels"),
    )

    # Structural (non-capacity) conflict families: IIS constraint-name prefix -> (short title, cause).
    # Most specific prefix first; a new structural rule adds one line.
    _STRUCTURAL_FAMILIES: tuple[tuple[str, str, str], ...] = (
        (
            "force_intermediate_reuse",
            "Fused intermediate must stay resident",
            "A fused group's intermediate is pinned on-chip and re-read rather than spilled to HBM (the AIE "
            "code-gen cannot stream partial results out and back), which fixes its reuse level across the fused loop.",
        ),
        (
            "force_output_reuse",
            "Output must stay resident",
            "An operator's output is pinned resident and reused in place (no partial spill to HBM), "
            "which fixes its reuse level.",
        ),
        (
            "zStop_Choose_One",
            "No consistent reuse schedule",
            "A tensor's reuse level (how long it stays resident) has no value consistent with the other pinned tensors "
            "-- typically a streamed (K-tiled) weight competing with a pinned activation for the same on-chip budget.",
        ),
    )

    def _structural_conflicts(self, names: list[str]) -> list[StructuralConflictIR]:
        """Group structural IIS constraint names into plain-language conflicts; unmatched names are left out."""
        buckets: dict[str, StructuralConflictIR] = {}
        for name in names:
            match = next((fam for fam in self._STRUCTURAL_FAMILIES if name.startswith(fam[0])), None)
            if match is None:
                continue
            key, title, explanation = match
            bucket = buckets.setdefault(key, StructuralConflictIR(title=title, explanation=explanation, constraints=[]))
            bucket.constraints.append(name)
        return list(buckets.values())

    # Per resource-capacity family: how to phrase and quantify its unmet inequality. `term_prefixes`
    # are the IIS constraint-name prefixes whose subject is a demand contributor for this family. A new
    # capacity family becomes quantifiable by adding one entry here + recording its bound/terms.
    _FAMILY_META: dict[str, dict[str, Any]] = {
        "memory_capacity": {
            "unit": "bytes",
            "demand_label": "tensors resident on",
            "bound_label": "on-chip memory of",
            "demand_input": "workload tensor sizes × mapping intra-core tiling",
            "bound_input": "hardware spec: core memory size",
            "term_prefixes": ("memload_", "u_eq_"),
            "term_detail": "min resident (1 tile)",
            "levers": (
                "Increase {core} on-chip memory (hardware spec)",
                "Tile the fused group finer so fewer / smaller tiles are resident (mapping intra_core_tiling)",
                "Reduce the workload's tensor sizes (fewer channels, smaller spatial, shorter sequence)",
            ),
        },
        "object_fifo_depth": {
            "unit": "FIFO slots",
            "demand_label": "concurrently buffered tiles on",
            "bound_label": "object-FIFO depth of",
            "demand_input": "mapping reuse levels × fused tiles",
            "bound_input": "hardware spec: tile object-FIFO depth",
            "term_prefixes": ("objfifo_", "u_eq_"),
            "term_detail": "min buffered tiles",
            "levers": (
                "Increase {core}'s object-FIFO depth (hardware spec)",
                "Lower buffering: reduce reuse / double-buffering (mapping)",
                "Route fewer tensors through {core} (mapping)",
            ),
        },
        "buffer_descriptors": {
            "unit": "descriptors",
            "demand_label": "buffer descriptors on",
            "bound_label": "buffer-descriptor budget of",
            "demand_input": "mapping tiling × transfers through the tile",
            "bound_input": "hardware spec: tile buffer-descriptor count",
            "term_prefixes": ("bddepth_", "u_eq_"),
            "term_detail": "min descriptors",
            "levers": (
                "Increase {core}'s buffer-descriptor budget (hardware spec)",
                "Reduce distinct transfers / reuse levels through {core} (mapping)",
                "Fuse fewer tensors through {core} (mapping)",
            ),
        },
    }

    def _limit_kind_from_name(self, name: str) -> str | None:
        """Map an untagged IIS constraint name to a resource-limit family (or None if it is a
        placement/indicator constraint that pins a resource but is not itself a hardware limit)."""
        return next((kind for prefix, kind in self._LIMIT_PREFIXES if name.startswith(prefix)), None)

    @staticmethod
    def _fmt_bytes(n: float) -> str:
        """Human-readable byte size for the diagnosis."""
        if abs(n) >= 1 << 20:
            return f"{n / (1 << 20):.2f} MB"
        if abs(n) >= 1 << 10:
            return f"{n / (1 << 10):.1f} KB"
        return f"{n:.0f} B"

    @staticmethod
    def _fmt_qty(value: float, unit: str) -> str:
        """Human-readable quantity for the diagnosis: bytes scale to KB/MB, counts stay integral."""
        if unit == "bytes":
            return TransferAndTensorAllocator._fmt_bytes(value)
        rounded = int(round(value))
        return f"{rounded} {unit}"

    @staticmethod
    def _term_value(info: Any) -> float:
        """A demand term is either a bare value or a ``{value, dims, dtype}`` record."""
        return float(info["value"]) if isinstance(info, dict) else float(info)

    @staticmethod
    def _term_meta(info: Any) -> tuple[list[tuple[str, int]], str]:
        """The tile shape (per-dim label+size) and dtype of a tensor demand term, if it carries them."""
        if isinstance(info, dict):
            return info.get("dims", []), info.get("dtype", "")
        return [], ""

    def _forced_terms(self, terms: dict[str, float], core_id: int, prefixes: tuple[str, ...], names: list[str]) -> dict:
        """The demand contributors the IIS actually forces onto this core: the exact subjects of its
        constraints whose prefix marks this family. Extracts the subject (up to the ``_Core_<id>``
        marker) rather than substring-matching, so a base tensor name is not double-counted with its
        partitions."""
        markers = (f"_Core_{core_id}", f"_Core {core_id}")
        forced: dict[str, float] = {}
        for cn in names:
            prefix = next((p for p in prefixes if cn.startswith(p)), None)
            if prefix is None:
                continue
            rest = cn[len(prefix) :]
            cut = min((rest.index(m) for m in markers if m in rest), default=-1)
            subject = rest[:cut] if cut >= 0 else rest
            if subject in terms:
                forced[subject] = terms[subject]
        return forced

    def _build_unmet(self, family: str, core_id: int, constraint_names: list[str]) -> UnmetConstraintIR | None:
        """The quantitative constraint the designer must satisfy for this resource family, as an
        intuitive ``demand <= bound`` inequality with per-contributor terms, input provenance, and the
        levers that would make it fit. Family-agnostic: driven by the recorded bound/terms + the family
        descriptor, so every recorded resource-capacity family (memory, object-FIFO, buffer descriptors)
        is quantified the same way. Uses only the terms the IIS forces onto this core."""
        meta = self._FAMILY_META.get(family)
        terms_all = self.context.ledger.terms.get((family, core_id))
        if meta is None or not terms_all:
            return None
        forced = self._forced_terms(terms_all, core_id, meta["term_prefixes"], constraint_names)
        return self._unmet_from_terms(family, core_id, forced)

    def _unmet_from_terms(self, family: str, core_id: int, forced: dict[str, Any]) -> UnmetConstraintIR | None:
        """Build the ``demand <= bound`` inequality for a family+core from an already-selected set of
        demand terms (used both by the IIS path and by the backend-agnostic capacity fallback)."""
        meta = self._FAMILY_META.get(family)
        bound = self.context.ledger.bounds.get((family, core_id))
        if meta is None or bound is None or not forced:
            return None
        unit = meta["unit"]
        demand = sum(self._term_value(v) for v in forced.values())
        gap = demand - bound
        core = f"Core {core_id}"
        terms = []
        for label, info in sorted(forced.items(), key=lambda kv: -self._term_value(kv[1])):
            dims, dtype = self._term_meta(info)
            terms.append(
                ConstraintTermIR(
                    label=label,
                    value=self._term_value(info),
                    detail=meta["term_detail"],
                    dtype=dtype,
                    dims=[TileDimIR(label=lbl, size=sz) for lbl, sz in dims],
                )
            )
        return UnmetConstraintIR(
            family=family,
            statement=(
                f"{meta['demand_label']} {core} need {self._fmt_qty(demand, unit)}, but the "
                f"{meta['bound_label']} {core} is {self._fmt_qty(bound, unit)} (short by {self._fmt_qty(gap, unit)})"
            ),
            demand_label=f"{meta['demand_label']} {core}",
            demand_value=demand,
            demand_input=meta["demand_input"],
            bound_label=f"{meta['bound_label']} {core}",
            bound_value=bound,
            bound_input=meta["bound_input"],
            operator="<=",
            gap=gap,
            unit=unit,
            terms=terms,
            levers=[lever.format(core=core) for lever in meta["levers"]],
        )

    def _build_unmet_direct(self, family: str, core_id: int) -> UnmetConstraintIR | None:
        """The unmet inequality from *all* recorded demand terms for a family+core, with no IIS
        filtering -- how the backend-agnostic fallback quantifies an over-budget resource."""
        terms_all = self.context.ledger.terms.get((family, core_id))
        if not terms_all:
            return None
        return self._unmet_from_terms(family, core_id, dict(terms_all))

    def _direct_capacity_overflows(self) -> list[ImplicatedResourceIR]:
        """Backend-agnostic infeasibility diagnosis: when the solver reports no IIS (e.g. OR-Tools), a
        per-core capacity is still over its budget in the recorded demand vs bound. Report each family
        whose recorded demand exceeds its *non-zero* hardware bound, most over-budget first, as the same
        quantitative unmet inequality the IIS path produces. A zero bound is a hardware concept that does
        not apply to this core (e.g. AIE object-FIFOs on a TPU-like core), so it is skipped -- the
        diagnosis highlights only real, actionable limits instead of every incidental constraint."""
        cores_by_id: dict[int, Core] = {
            res.id: res for _k, res in self.context.ledger.constraints.values() if isinstance(res, Core)
        }
        overflows: list[tuple[float, str, int]] = []
        for (family, core_id), terms in self.context.ledger.terms.items():
            bound = self.context.ledger.bounds.get((family, core_id))
            if not bound or bound <= 0:
                continue
            demand = sum(self._term_value(v) for v in terms.values())
            if demand > bound:
                overflows.append((demand - bound, family, core_id))
        overflows.sort(reverse=True)  # biggest overshoot first, so the worst offender leads the report
        resources: list[ImplicatedResourceIR] = []
        for _gap, family, core_id in overflows:
            core = cores_by_id.get(core_id)
            ref = (
                self._resource_ref_ir(core)
                if core is not None
                else ResourceRefIR(kind="core", id=str(core_id), label=f"Core {core_id}")
            )
            resources.append(
                ImplicatedResourceIR(
                    resource=ref,
                    constraint_kinds=[family],
                    reason=self._KIND_REASON.get(family, family.replace("_", " ")),
                    constraints=[],
                    unmet=self._build_unmet_direct(family, core_id),
                )
            )
        return resources

    def _resolve_iis_constraint(self, name: str) -> tuple[ResourceRefIR | None, str | None]:
        """Resolve one IIS constraint name to (physical-resource ref, resource-limit family). A tagged
        constraint yields a precise, detailed ref; otherwise a core id is recovered from the name
        (``Core 3`` / sanitized ``Core_3``). Returns ``(None, None)`` for a structural constraint."""
        tag = self.context.ledger.constraints.get(name)
        if tag is not None:
            kind, resource = tag
            return self._resource_ref_ir(resource), kind
        match = re.search(r"Core[ _](\d+)", name)
        if match:
            cid = match.group(1)
            return ResourceRefIR(kind="core", id=cid, label=f"Core {cid}"), self._limit_kind_from_name(name)
        return None, None

    def _resource_ref_ir(self, resource: Resource) -> ResourceRefIR:
        """Serialize a physical resource to the IR the architecture view highlights. Extend with new
        ``isinstance`` arms as new resource kinds get bound to constraints."""
        if isinstance(resource, Core):
            detail: dict[str, str] = {"core_type": str(resource.core_type)}
            try:
                detail["memory_capacity_bits"] = str(resource.get_memory_capacity())
            except Exception:  # noqa: BLE001 -- a missing capacity must not break the diagnosis
                pass
            return ResourceRefIR(kind="core", id=str(resource.id), label=f"Core {resource.id}", detail=detail)
        return ResourceRefIR(kind="link", id=_resource_key(resource), label=_resource_key(resource))

    def _implicated_resource(self, entry: dict[str, Any]) -> ImplicatedResourceIR:
        """Build the diagnosis for one implicated resource: its reason and -- when a recorded capacity
        family (memory / object-FIFO / buffer descriptors) is among its conflicts -- the quantitative
        unmet inequality."""
        kinds = sorted(entry["kinds"])
        ref = entry["ref"]
        unmet: UnmetConstraintIR | None = None
        if ref.kind == "core" and ref.id.isdigit():
            for fam in kinds:  # quantify the first family with recorded bound + demand
                unmet = self._build_unmet(fam, int(ref.id), entry["constraints"])
                if unmet is not None:
                    break
        # gap > 0 is a real overflow (report the limit it broke). A non-positive gap means the IIS pulled
        # this capacity constraint into a structural conflict without the core being over budget ("involved", not over).
        overflow = unmet is not None and unmet.gap > 0
        capacity_kinds = self._FAMILY_META.keys() | {"memory_capacity"}
        if overflow:
            reason = "; ".join(self._KIND_REASON.get(k, k.replace("_", " ")) for k in kinds)
        else:
            unmet = None
            if kinds and all(k in capacity_kinds for k in kinds):
                reason = "involved in a reuse/scheduling conflict (not over budget)"
            elif kinds:
                reason = "; ".join(self._KIND_REASON.get(k, k.replace("_", " ")) for k in kinds)
            else:
                reason = "conflicting allocation constraints"
        return ImplicatedResourceIR(
            resource=ref, constraint_kinds=kinds, reason=reason, constraints=entry["constraints"], unmet=unmet
        )

    def _build_infeasibility_report(self, status: str, group: str | None = None) -> InfeasibilityReportIR:
        """Turn an infeasible solve into a per-resource diagnosis. Uses the solver's IIS (Gurobi) to
        get the minimal conflicting constraint set, maps each back to its physical resource via the
        registry (falling back to a ``Core N`` name match for any untagged per-core constraint), and
        groups the result so a consumer can highlight the offending cores/links with a reason."""
        stats = self.model.solve_stats()
        iis_available = self.model.supports_iis
        iis_names: list[str] = []
        if iis_available:
            try:
                self.model.compute_iis()
                iis_names = self.model.iis_constraints()
            except Exception as exc:  # noqa: BLE001 -- IIS is best-effort diagnostics
                _logger.warning(f"IIS computation failed: {exc}")
                iis_available = False

        grouped: dict[str, dict[str, Any]] = {}
        unbound: list[str] = []
        for name in iis_names:
            ref, kind = self._resolve_iis_constraint(name)
            if ref is None:
                unbound.append(name)
                continue
            entry = grouped.setdefault(f"{ref.kind}:{ref.id}", {"ref": ref, "kinds": set(), "constraints": []})
            if ref.detail and not entry["ref"].detail:
                entry["ref"] = ref  # prefer the detailed (tagged) ref so the tooltip has capacity facts
            if kind:
                entry["kinds"].add(kind)
            entry["constraints"].append(name)

        resources = [self._implicated_resource(entry) for entry in grouped.values()]
        # Backend-agnostic fallback: if the IIS pinned nothing to a resource (or is unavailable, e.g. on
        # OR-Tools), diagnose directly from the recorded per-core demand vs bound so an OR-Tools solve
        # still yields the same actionable per-resource diagnosis instead of a bare "infeasible".
        if not resources:
            resources = self._direct_capacity_overflows()

        # Group structural constraints (unbound + overflow-less resources) into plain-language causes.
        structural_names = list(unbound) + [c for r in resources if r.unmet is None for c in r.constraints]
        conflicts = self._structural_conflicts(structural_names)

        quantified = next((r for r in resources if r.unmet is not None), None)
        involved_cores = ", ".join(r.resource.label for r in resources if r.resource.kind == "core")
        if quantified is not None:
            nature = "capacity"
            summary = f"Infeasible mapping — {quantified.resource.label}: {quantified.unmet.statement}"
        elif conflicts:
            nature = "structural"
            lead = conflicts[0]
            summary = f"Infeasible mapping — structural conflict: {lead.title.lower()}. {lead.explanation}" + (
                f" Involves {involved_cores}." if involved_cores else ""
            )
        elif resources:
            nature = "unknown"
            summary = f"Infeasible mapping: {resources[0].reason}" + (f" on {involved_cores}" if involved_cores else "")
        elif not iis_available:
            nature = "unknown"
            summary = (
                f"Infeasible mapping ({status}); no single per-core capacity is over its recorded budget, "
                f"so the conflict is structural (e.g. transfer routing). The {stats.backend} backend cannot "
                "compute an IIS -- re-run with the Gurobi backend for the minimal conflict set."
            )
        else:
            nature = "unknown"
            summary = (
                f"Infeasible mapping ({status}); {len(iis_names)} conflicting constraints, "
                "none pinned to a single resource."
            )

        return InfeasibilityReportIR(
            status=status,
            backend=stats.backend,
            solver=stats.solver,
            group=group,
            iis_available=iis_available,
            nature=nature,
            resources=resources,
            conflicts=conflicts,
            unbound_constraints=unbound,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # public solve()                                                     #
    # ------------------------------------------------------------------ #
    def solve(self, *, time_limit_s: float, tee: bool = False, total_mac_ops: int | None = None) -> AllocationSolution:
        """Solve the model within ``time_limit_s`` and read the allocation back; ``tee`` prints the solver log, and
        ``total_mac_ops`` of the untiled group adds its end-to-end MAC utilization to the performance report."""
        self.model.set_param(SolverParams.VERBOSITY, 1 if tee else 0)
        self.model.set_param(SolverParams.TIME_LIMIT, time_limit_s)
        with span("milp_optimize"):
            self.model.optimize(self._mip_progress_callback)
        status = self.model.get_status()
        if status == "TIME_LIMIT" and self.model.get_sol_count() > 0:
            _logger.warning(
                "Allocation solve hit the %ss limit; taking the best incumbent (gap unknown)",
                time_limit_s,
            )
        elif status != "OPTIMAL":
            # Produce a structured, per-resource diagnosis (this computes the IIS on backends that
            # support it) instead of a bare failure, so a launch with an invalid mapping still yields
            # an inspectable result. The .ilp (Gurobi's IIS) is still written for offline debugging.
            report = self._build_infeasibility_report(self.model.get_status())
            try:
                os.makedirs(self.output_path, exist_ok=True)
                self.model.write(os.path.join(self.output_path, "model.ilp"))
            except Exception:  # noqa: BLE001 -- .ilp export is best-effort (Gurobi-only)
                pass
            raise InfeasibleAllocationError(report)

        with span("milp_extract"):
            tensor_alloc = self.get_tensor_allocations()
            routing = self.get_transfer_routing()
            chosen_memory_cores = self.get_chosen_memory_cores()
            tensor_reuse_levels = self.get_tensor_reuse_levels()
            tensor_depths = self.get_tensor_depths()

        with span("milp_report"):
            value, q = self.model.value, self.quantities
            latency = Latency(
                total=int(value(q.get("total_latency").expr)),
                per_iteration=int(sum(slot_lat.X for slot_lat in self.vars.slot_latency.values())),
                overlap=int(value(q.get("overlap").expr)),
                fill=round(value(q.get("fill").expr)),
            )
            return AllocationSolution(
                tensor_placements=tensor_alloc,
                transfer_routes=routing,
                memory_cores=chosen_memory_cores,
                reuse_levels=tensor_reuse_levels,
                depths=tensor_depths,
                single_buffered=frozenset(self.get_single_buffered()),
                latency=latency,
                primary_cost=self.primary_cost(),
                throughput_bound=self.throughput_bound(),
                solve_stats=self.model.solve_stats(),
                performance=self._performance_report(latency.total, total_mac_ops),
                capacity_slack=self._reported_capacity_slack(),
            )

    def _performance_report(self, total_latency: int, total_mac_ops: int | None) -> dict[str, Any] | None:
        """:meth:`compute_performance_stats` with the end-to-end MAC utilization; None if it fails."""
        try:
            performance = self.compute_performance_stats()
        except Exception as exc:  # observability must never break the solve
            _logger.warning("Failed to compute performance stats: %s", exc)
            return None
        try:
            performance["aggregate"] |= end_to_end_mac_utilization(self.space.accelerator, total_mac_ops, total_latency)
        except Exception as exc:
            _logger.warning("Failed to compute end-to-end MAC utilization: %s", exc)
        return performance

    def _reported_capacity_slack(self) -> dict[int, dict[str, float]]:
        try:
            return self.capacity_slack()
        except Exception as exc:
            _logger.warning("Failed to compute capacity slack: %s", exc)
            return {}

    def get_tensor_reuse_levels(self) -> TensorReuseLevels:
        reuse_levels: TensorReuseLevels = {}
        for t in self.space.workload.tensors:
            for stop in self.space.stops(t):
                if self.vars.z_stop[(t, stop)].X > self.VAR_THRESHOLD:
                    reuse_levels[t] = stop
        return reuse_levels

    def get_single_buffered(self) -> set[Tensor]:
        """The tensors whose moving window the solve holds in one buffer."""
        return {t for (t, _), z in self.vars.z_single.items() if z.X > self.VAR_THRESHOLD}

    def get_tensor_depths(self) -> TensorDepths:
        tiles_needed: TensorDepths = {}
        for t in self.space.workload.tensors:
            for stop in self.space.stops(t):
                if self.vars.z_stop[(t, stop)].X > self.VAR_THRESHOLD:
                    tiles_needed[t] = self.space.tiles_needed_levels[(t, stop)]
        return tiles_needed

    def get_transfer_routing(self) -> TransferAlloc:
        routing: TransferAlloc = {}
        for tr in self.space.transfer_nodes:
            chosen = [
                choice for choice in self.space.path_choices[tr] if self.vars.y[(tr, choice)].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{tr.name}: expected exactly one routing choice, got {chosen}")
            routing[tr] = chosen[0]
        return routing

    def get_chosen_memory_cores(self) -> MemoryAlloc:
        chosen_memory_cores: MemoryAlloc = {}
        tensor_alloc = self.get_tensor_allocations()
        for tr in self.space.transfer_nodes:
            if not self.space.is_const_io(tr):
                continue
            tensor = self.space.constant_transfer_tensor(tr)
            if tensor in tensor_alloc:
                chosen_memory_cores[tr] = tensor_alloc[tensor]
            else:
                chosen_memory_cores[tr] = self.space.fixed_choice(tensor)
        return chosen_memory_cores

    def get_tensor_allocations(self) -> TensorAlloc:
        tensor_alloc: TensorAlloc = {}
        for t in self.space.tensor_fixed:
            tensor_alloc[t] = self.space.fixed_choice(t)
        for t in self.space.tensor_var:
            chosen = [
                choice for choice in self.space.tensor_choices[t] if self.vars.x[(t, choice)].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{t.node_name}: expected exactly one placement choice, got {chosen}")
            tensor_alloc[t] = chosen[0]
        return tensor_alloc

    def _mip_progress_callback(self, model, where):
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
            self.optimization_trace.append(point)
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

        self.optimization_trace.append(point)

    def save_optimization_metrics(self, save_path: str) -> None:
        """
        Dump a concise YAML summary of the Gurobi run alongside the progress plot.

        Headline fields (in order of relevance for a paper):
          - search:    nodes explored, simplex/barrier iterations  -> "how much was searched"
          - solution:  objective, best bound, MIP gap              -> "what was found / how tight"
          - effort:    runtime (s), work units                     -> "how expensive it was"
          - model:    variable / constraint / nonzero counts      -> "problem size"
          - trace:     per-event progress records (reuses self.optimization_trace)
        """

        def _attr(name: str) -> Any | None:
            """Return a Gurobi model attribute or None if unavailable post-solve."""
            # Access the underlying gurobipy model for Gurobi-specific attributes.
            # GurobiBackend stores the model as ._model; fall back gracefully for other backends.
            from stream.opt.solver import GurobiBackend  # noqa: PLC0415

            raw_model = self.model._model if isinstance(self.model, GurobiBackend) else None
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
        for rec in self.optimization_trace:
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
        _logger.info("Optimization metrics saved to %s", save_path)

    @staticmethod
    def _json_scalar(v: Any) -> Any:
        """Coerce a solver/cost value into a JSON-safe scalar (int when integral, else float)."""
        if v is None or isinstance(v, bool | str):
            return v
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if not math.isfinite(f):
            return str(v)
        return int(f) if f.is_integer() else f

    def compute_performance_stats(self) -> dict[str, Any]:
        """Read-only performance summary of the solved schedule.

        Derives -- WITHOUT changing any cost or latency model -- a structured view of
        where the schedule's latency goes, so callers (and AI agents via the IR
        performance view) can reason about *utilization*, not just total latency:

          * per compute node: how many cores it is inter-core-tiled across, its latency
            contribution, the ideal (perfect-spatial-utilization) compute cycles, the
            MAC spatial utilization, and the compute efficiency (ideal / actual);
          * the per-iteration latency split into compute-bound vs transfer-bound cycles
            (which resource class sets each slot's latency);
          * aggregate utilization (cores used vs available, latency-weighted MAC util).

        Every value comes straight from the cost LUT (CostModelEvaluation) and the
        solved slot latencies; this method is purely observational.
        """

        space = self.space

        def _node_active(n: Node) -> tuple[int, list[Core]]:
            cores = space.cost_lut.get_cores(n)
            runtime = ceil(max(space.cost_lut.get_cost(n, c).latency_total for c in cores)) if cores else 0
            return get_active_latency(n, float(runtime), space.ssis), cores

        # ── Per compute-node utilization ── #
        per_node: dict[str, dict[str, Any]] = {}
        for n in space.ssc_nodes:
            try:
                active, cores = _node_active(n)
                if not cores:
                    continue
                entry = space.cost_lut.get_cost(n, cores[0])
                ideal = getattr(entry, "ideal_cycle", None)
                mac_util = getattr(entry, "mac_spatial_utilization", None)
                efficiency = (float(ideal) / active) if (ideal and active) else None
                # "Degenerate" = a matmul/conv node whose ZigZag estimate fell back to the
                # 1-MAC/cycle scalar cost (cme is None): the spatial array was not modelled, so
                # the latency is untrustworthy (typically orders of magnitude too high). Activation
                # / elementwise ops (silu, mul, ...) also report cme None but their scalar estimate
                # is legitimate, so they are NOT counted as degenerate.
                node_type = str(getattr(getattr(entry, "layer", None), "type", "")).lower()
                is_mac = any(k in node_type for k in ("conv", "gemm", "matmul", "linear"))
                fallback = bool(getattr(entry, "cme", None) is None and is_mac)
                per_node[getattr(n, "name", str(n))] = {
                    "kind": "compute",
                    "n_cores": len(cores),
                    "latency_cycles": int(active),
                    "ideal_compute_cycles": self._json_scalar(ideal),
                    "mac_spatial_utilization": self._json_scalar(mac_util),
                    "compute_efficiency": self._json_scalar(efficiency),
                    "fallback": fallback,
                }
            except Exception:
                continue

        # ── Per-iteration latency split by the resource class that sets each slot ── #
        compute_active_by_slot: dict[int, float] = {}
        for n in space.ssc_nodes:
            try:
                active, _ = _node_active(n)
                s = int(space.slot_of[n])
                compute_active_by_slot[s] = max(compute_active_by_slot.get(s, 0.0), float(active))
            except Exception:
                continue
        compute_cycles = transfer_cycles = 0.0
        for s, lat_var in self.vars.slot_latency.items():
            try:
                lat = float(lat_var.X)
            except Exception:
                continue
            if lat <= 0:
                continue
            if compute_active_by_slot.get(int(s), 0.0) >= lat * 0.999:
                compute_cycles += lat
            else:
                transfer_cycles += lat
        per_iter = compute_cycles + transfer_cycles
        bottleneck = {
            "compute_bound_cycles": int(compute_cycles),
            "transfer_bound_cycles": int(transfer_cycles),
            "compute_bound_pct": round(100.0 * compute_cycles / per_iter, 2) if per_iter else None,
            "transfer_bound_pct": round(100.0 * transfer_cycles / per_iter, 2) if per_iter else None,
        }

        # ── Aggregate utilization ── #
        nodes = list(per_node.values())
        total_active = sum(d["latency_cycles"] for d in nodes) or 1
        weighted_util = sum((d["mac_spatial_utilization"] or 0.0) * d["latency_cycles"] for d in nodes) / total_active
        utils = [d["mac_spatial_utilization"] for d in nodes if d["mac_spatial_utilization"] is not None]
        cores_used: set[int] = set()
        for n in space.ssc_nodes:
            for c in space.cost_lut.get_cores(n) or []:
                cores_used.add(c.id)
        offchip_id = space.accelerator.offchip_core_id
        degenerate_nodes = [name for name, d in per_node.items() if d.get("fallback")]
        aggregate = {
            "compute_cores_available": sum(1 for c in space.accelerator.core_list if c.id != offchip_id),
            "compute_cores_used": len(cores_used),
            "latency_weighted_mac_spatial_utilization": self._json_scalar(weighted_util),
            "min_mac_spatial_utilization": self._json_scalar(min(utils) if utils else None),
            # True iff a matmul/conv node fell back to the scalar cost (latency untrustworthy).
            "degenerate": bool(degenerate_nodes),
            "degenerate_nodes": degenerate_nodes,
        }

        return {
            "per_node": per_node,
            "bottleneck": bottleneck,
            "aggregate": aggregate,
            "overlap": self._overlap_section(),
            "tensor_reuse": self._tensor_reuse_breakdown(),
            "memory_occupancy": self._memory_occupancy(),
        } | self._family_reports()

    def _family_reports(self) -> dict[str, Any]:
        reports: dict[str, Any] = {}
        for family in self.families.families:
            if isinstance(family, ReportingFamily):
                reports |= family.report(self.context, self.quantities)
        return reports

    def primary_cost(self) -> float:
        """The solved value of the latency objective, whichever lexicographic level the backend ended on."""
        return self.model.value(self.objective["latency"].expr)

    def throughput_bound(self) -> float:
        """The pipelined compute bound of the steady state, from the solved allocation."""
        space, q, value = self.space, self.quantities, self.model.value
        busy: dict[Any, float] = defaultdict(float)
        for n in space.ssc_nodes:
            latencies = [space.cost_lut.get_cost(n, c).latency_total for c in space.cost_lut.get_cores(n)]
            runtime = ceil(max(latencies)) if latencies else 0
            active = float(get_active_latency(n, float(runtime), space.ssis))
            for group in space.mapping.get(n).resource_allocation:
                for core in group:
                    busy[core] += active
        per_iteration = max(busy.values(), default=0.0)
        per_iteration = max(per_iteration, float(q.get("recurrence_bound").expr))
        for shared in q.indexed("shared_busy").values() if "shared_busy" in q else ():
            per_iteration = max(per_iteration, value(shared.expr))
        chain = sum(float(v.X) for v in self.vars.slot_latency.values())
        return space.iterations * per_iteration + max(0.0, chain - per_iteration) + value(q.get("fill").expr)

    def capacity_slack(self) -> dict[int, dict[str, float]]:
        """Unused capacity per core: memory in bytes, fifo depth and buffer descriptors in slots."""
        slack: dict[int, dict[str, float]] = {}
        for row in self._memory_occupancy():
            slack.setdefault(row["core_id"], {})["memory_bytes"] = (row["capacity_bits"] - row["resident_bits"]) / 8
        ledger = self.context.ledger
        for (family, core_id), terms in ledger.loads.items():
            bound = ledger.bounds.get((family, core_id))
            if bound is None:
                continue
            try:
                used = sum(count for var, count in terms if float(var.X) > self.VAR_THRESHOLD)
            except Exception:  # noqa: BLE001
                continue
            slack.setdefault(core_id, {})[family] = bound - used
        return slack

    def _memory_occupancy(self) -> list[dict[str, Any]]:
        """Per core: bits the solved placement keeps resident vs capacity (from the memory-capacity constraint)."""
        rows: list[dict[str, Any]] = []
        ledger = self.context.ledger
        cores = {c.id: c for c in self.space.accelerator.core_list}
        for core_id, terms in sorted(ledger.memory.items()):
            core = cores.get(core_id)
            if core is None:
                continue
            handed = ledger.handover_bits.get(core_id, 0)
            per_tensor: dict[str, int] = defaultdict(int)
            try:
                for indicator, bits, tensor_name in terms:
                    # A MILP binary comes back as 0.9999...; anything above the midpoint is a 1.
                    if float(indicator.X) > self.VAR_THRESHOLD:
                        per_tensor[tensor_name] += bits
            except Exception:  # noqa: BLE001 -- an unreadable solution means no measurement, not zero
                continue
            # A copy within one memory holds what it needs beyond its source, and never less than nothing.
            per_tensor = {name: bits for name, bits in per_tensor.items() if bits > 0}
            if handed:
                per_tensor["handover"] = handed
            resident = sum(per_tensor.values())
            try:
                bound = ledger.bounds.get(("memory_capacity", core_id))
                capacity = int(bound * 8) if bound is not None else int(core.get_memory_capacity())
            except Exception:  # noqa: BLE001
                continue
            rows.append(
                {
                    "core_id": core_id,
                    "core_name": str(getattr(core, "type", "")) or str(core),
                    "resident_bits": resident,
                    "capacity_bits": capacity,
                    "utilization": (resident / capacity) if capacity else None,
                    # Largest contributors first: the tensors that set the floor on any shrink.
                    "tensors": [
                        {"tensor": name, "bits": bits}
                        for name, bits in sorted(per_tensor.items(), key=lambda kv: -kv[1])[:_OCCUPANCY_TOP_TENSORS]
                    ],
                }
            )
        return rows

    def _tensor_reuse_breakdown(self) -> list[dict[str, Any]]:
        """Per-tensor on-chip reuse chosen by the solver -- to make the fused execution legible.

        For each tensor: how many steady-state iterations it stays resident on-chip
        (``reuse_factor``; 1 means it is re-fetched every iteration, e.g. streamed weights), the loop
        level reuse stops at (``reuse_stop_level``; -1 = none), the tile buffers that residency needs
        (``on_chip_tiles``), the tensor size, and its steady-state loop nest (outermost -> innermost;
        each loop shows type S/K/ST/T and effect V=varying / I=invariant / A=absent). A large tensor
        with ``reuse_factor == 1`` is the signature of a memory-bandwidth-bound fused schedule -- e.g.
        gemm weights that the solver chose to re-stream from DRAM every iteration. Sorted largest
        first. Purely observational.
        """
        try:
            stops = self.get_tensor_reuse_levels()
        except Exception:  # noqa: BLE001
            return []
        rows: list[dict[str, Any]] = []
        for t, ssis in self.space.ssis.items():
            if not isinstance(t, Tensor):
                continue
            stop = stops.get(t)
            try:
                size_bits = int(t.size_bits())
            except Exception:  # noqa: BLE001
                size_bits = None
            rows.append(
                {
                    "tensor": getattr(t, "name", str(t)),
                    "size_bits": size_bits,
                    "reuse_factor": self.space.reuse_levels.get((t, stop)) if stop is not None else None,
                    "reuse_stop_level": stop,
                    "on_chip_tiles": self.space.tiles_needed_levels.get((t, stop)) if stop is not None else None,
                    "loop_nest_out_to_in": [repr(v) for v in reversed(ssis.variables)],
                }
            )
        rows.sort(key=lambda d: -(d["size_bits"] or 0))
        return rows

    def _overlap_section(self) -> dict[str, Any]:
        """Overlap summary: the inter-iteration overlap, which resources bind it (those at the MINIMUM
        slack, the resource-side cap), the per-resource slack, and the recurrence bound (RecMII; 0 for
        feed-forward) that separately caps it."""
        slack = self._resource_slack_breakdown()
        q = self.quantities
        try:
            overlap_cycles = int(self.model.value(q.get("overlap").expr)) if "overlap" in q else None
        except Exception:  # noqa: BLE001
            overlap_cycles = None
        min_slack = min((d["slack_cycles"] for d in slack), default=None)
        binding = [d["resource"] for d in slack if d["slack_cycles"] == min_slack]
        return {
            "overlap_cycles": overlap_cycles,
            "binding_resources": binding,
            "per_resource_slack": slack,
            "recurrence_bound_cycles": q.get("recurrence_bound").expr if "recurrence_bound" in q else 0,
        }

    def _resource_slack_breakdown(self) -> list[dict[str, Any]]:
        """Per-resource steady-state slack (boundary idle within one iteration), ascending.

        The TETRA inter-iteration overlap equals the MINIMUM slack across all resources (compute
        cores AND communication links): a resource busy from an early to a late slot has zero
        boundary idle and therefore pins the overlap to zero. Sorted ascending so the binding
        resource(s) come first. Purely observational.
        """
        rows: list[dict[str, Any]] = []
        idle = self.quantities.indexed("idle_latency") if "idle_latency" in self.quantities else {}
        for res, quantity in idle.items():
            try:
                slack = int(round(self.model.value(quantity.expr)))
            except Exception:
                continue
            rows.append(
                {
                    "resource": str(res),
                    "kind": "core" if isinstance(res, Core) else "link",
                    "slack_cycles": slack,
                }
            )
        rows.sort(key=lambda d: d["slack_cycles"])
        return rows

    def save_slot_latency_breakdown(self, save_path: str) -> None:  # noqa: PLR0915, PLR0912
        """Dump a debug-friendly per-slot latency breakdown next to the metrics yaml.

        For every slot lists the compute and transfer contributors with the
        intermediate values that make up its slot_latency constraint:
          - compute: LUT latency_total, SSIS fraction, active_latency
          - transfer: tensor bits, min link bandwidth, raw path cycles,
                      active_latency (absent-loop scaled), reuse_factor,
                      final contribution to slot_latency

        Best-effort: any per-node failure is silently skipped, and the whole
        method swallows top-level errors so it never blocks the pipeline.
        """

        def _scalar(v: Any) -> Any:
            if v is None or isinstance(v, bool | str):
                return v
            try:
                f = float(v)
            except (TypeError, ValueError):
                return str(v)
            if not math.isfinite(f):
                return str(v)
            if f.is_integer():
                return int(f)
            return f

        space, q, value = self.space, self.quantities, self.model.value
        try:
            breakdown: dict[int, dict[str, Any]] = {}
            for s, lat_var in self.vars.slot_latency.items():
                slot_val: float | None
                try:
                    slot_val = float(lat_var.X)
                except Exception:
                    slot_val = None
                breakdown[int(s)] = {
                    "slot_latency_cycles": _scalar(slot_val),
                    "compute_contributors": [],
                    "transfer_contributors": [],
                }

            # ── Compute contributors ── #
            for n in space.ssc_nodes:
                try:
                    s = int(space.slot_of[n])
                    cores = space.cost_lut.get_cores(n)
                    latencies = [space.cost_lut.get_cost(n, c).latency_total for c in cores]
                    runtime = ceil(max(latencies)) if latencies else 0
                    active = get_active_latency(n, float(runtime), space.ssis)
                    try:
                        fraction: float | None = active_fraction(n, space.ssis)
                    except Exception:
                        fraction = None
                    breakdown.setdefault(
                        s,
                        {"slot_latency_cycles": None, "compute_contributors": [], "transfer_contributors": []},
                    )
                    breakdown[s]["compute_contributors"].append(
                        {
                            "name": getattr(n, "name", str(n)),
                            "n_cores_in_lut": len(cores),
                            "lut_latency_total": _scalar(runtime),
                            "ssis_fraction": _scalar(fraction),
                            "active_latency": _scalar(active),
                        }
                    )
                except Exception:
                    continue

            # ── Transfer contributors (only the chosen path per transfer) ── #
            for (tr, choice), y in self.vars.y.items():
                try:
                    if float(y.X) < 0.5:  # noqa: PLR2004
                        continue
                    s = int(space.slot_of[tr])
                    raw = int(space.transfer_latency_for_path(tr, choice))
                    active_abs = int(get_active_latency(tr, float(raw), space.ssis))
                    try:
                        reuse_factor = value(q.get("reuse_factor", tr).expr)
                    except Exception:
                        reuse_factor = None
                    try:
                        contribution = value(q.get("transfer_latency", (tr, choice)).expr)
                    except Exception:
                        contribution = None
                    tensor_bits: int | None = None
                    try:
                        if tr.inputs:
                            tensor_bits = int(tr.inputs[0].size_bits())
                    except Exception:
                        tensor_bits = None
                    min_bw: int | None = None
                    try:
                        if choice and choice.links_used:
                            min_bw = int(min(link.bandwidth for link in choice.links_used))
                    except Exception:
                        min_bw = None
                    breakdown.setdefault(
                        s,
                        {"slot_latency_cycles": None, "compute_contributors": [], "transfer_contributors": []},
                    )
                    breakdown[s]["transfer_contributors"].append(
                        {
                            "name": getattr(tr, "name", str(tr)),
                            "tensor_bits": _scalar(tensor_bits),
                            "min_link_bw": _scalar(min_bw),
                            "raw_path_cycles": _scalar(raw),
                            "active_latency_absent_loops": _scalar(active_abs),
                            "reuse_factor": _scalar(reuse_factor),
                            "contribution": _scalar(contribution),
                        }
                    )
                except Exception:
                    continue

            # ── Top-level totals ── #
            try:
                latency_per_iteration = sum(float(v.X) for v in self.vars.slot_latency.values())
            except Exception:
                latency_per_iteration = None
            try:
                overlap_val = value(q.get("overlap").expr)
            except Exception:
                overlap_val = None
            try:
                total_latency_val = value(q.get("total_latency").expr)
            except Exception:
                total_latency_val = None
            iter_step_val: float | None
            if latency_per_iteration is not None and overlap_val is not None:
                iter_step_val = latency_per_iteration - overlap_val
            else:
                iter_step_val = None

            summary = {
                "totals": {
                    "latency_per_iteration": _scalar(latency_per_iteration),
                    "overlap": _scalar(overlap_val),
                    "iter_step": _scalar(iter_step_val),
                    "total_latency": _scalar(total_latency_val),
                    "shared_busy": {
                        core: _scalar(value(busy.expr))
                        for core, busy in (q.indexed("shared_busy") if "shared_busy" in q else {}).items()
                    },
                },
                # Per-resource slack; the overlap equals the minimum (the binding resource(s) first).
                "resource_slack": self._resource_slack_breakdown(),
                # Per-tensor on-chip reuse (reuse_factor 1 = re-streamed every iteration).
                "tensor_reuse": self._tensor_reuse_breakdown(),
                "slots": [{"slot": s, **breakdown[s]} for s in sorted(breakdown)],
            }

            with open(save_path, "w") as fh:
                yaml.safe_dump(summary, fh, sort_keys=False, default_flow_style=False)
            _logger.info("Slot latency breakdown saved to %s", save_path)
        except Exception as e:
            # Never block the pipeline on this debug artifact.
            _logger.warning("save_slot_latency_breakdown failed: %s", e)

    def plot_optimization_progress(  # noqa: PLR0912, PLR0915
        self,
        *,
        save_path: str | None = None,
        show: bool = True,
        figsize: tuple[float, float] = (10.0, 6),
        show_work_subplot: bool = False,
    ) -> None:
        """
        Plot optimization progress recorded in self.optimization_trace.

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

        if not hasattr(self, "optimization_trace") or not self.optimization_trace:
            # Non-Gurobi backends do not populate optimization_trace via the callback.
            # Silently skip plotting rather than raising so OR-Tools solves succeed.
            return

        def _is_finite_number(x) -> bool:
            return x is not None and isinstance(x, int | float) and math.isfinite(x)

        trace = [
            rec
            for rec in self.optimization_trace
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
            _logger.info("Optimization progress plot saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_optimization_trace(self, file_path: str) -> None:
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
        if not hasattr(self, "optimization_trace") or not self.optimization_trace:
            # Non-Gurobi backends do not populate optimization_trace via the callback.
            # Silently skip saving rather than raising so OR-Tools solves succeed.
            return

        def _fin(x) -> bool:
            return x is not None and isinstance(x, int | float) and math.isfinite(x)

        # Sort by time, MIPSOL first when times are equal (mirrors plot logic)
        sorted_trace = sorted(
            self.optimization_trace,
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

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump({"trace": entries}, f, default_flow_style=False, sort_keys=False)

        _logger.info("Optimization trace saved to %s", file_path)
