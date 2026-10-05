"""Why an allocation model has no solution, as an inspectable report rather than a bare failure: each constraint is
created with what it stands for -- the resource it binds, the hardware limit it is part of, the tensor whose demand it
carries, or the modelling rule it enforces -- so the solver's IIS maps back to cores, links and causes."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from stream.hardware.architecture.core import Core
from stream.ir.infeasibility import (
    ConstraintTermIR,
    ImplicatedResourceIR,
    InfeasibilityReportIR,
    ResourceRefIR,
    StructuralConflictIR,
    TileDimIR,
    UnmetConstraintIR,
)
from stream.opt.allocation.constraint_optimization.utils import resource_key

if TYPE_CHECKING:
    from stream.mapping.mapping import Resource
    from stream.opt.allocation.constraint_optimization.formulation import ResourceLedger
    from stream.opt.solver import SolverModel

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceKind:
    """A hardware limit constraints bind, and how a diagnosis states it: ``reason`` is what exceeding it means; a
    kind with a ``unit`` is quantified as its demand against its bound, with the ``levers`` that would make it fit."""

    name: str
    reason: str
    unit: str = ""
    demand_label: str = ""
    bound_label: str = ""
    demand_input: str = ""
    bound_input: str = ""
    term_detail: str = ""
    levers: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class StructuralRule:
    """A modelling rule that is no hardware limit, and how a diagnosis explains it when it is part of a conflict."""

    title: str
    explanation: str


class ConstraintTag(NamedTuple):
    """What one constraint stands for: the ``resource`` it binds or is attributed to, the hardware limit ``kind``
    it is part of, the tensor ``subject`` whose demand on the resource it carries, or the ``rule`` it enforces."""

    resource: Resource | None = None
    kind: ResourceKind | None = None
    subject: str | None = None
    rule: StructuralRule | None = None


def structural_infeasibility(reason: str, model: SolverModel | None = None) -> InfeasibilityReportIR:
    """A minimal infeasibility report for a structural problem in the mapping itself (a node with no
    valid core), raised during model construction -- so an unbuildable model fails with an
    inspectable diagnosis rather than a bare exception."""
    backend = solver = "n/a"
    if model is not None:
        with suppress(AttributeError):
            stats = model.solve_stats()
            backend, solver = stats.backend, stats.solver
    return InfeasibilityReportIR(
        status="INFEASIBLE",
        backend=backend,
        solver=solver,
        group=None,
        iis_available=False,
        nature="structural",
        resources=[],
        unbound_constraints=[reason],
        summary=(
            f"Infeasible mapping: {reason}. The auto-generated mapping could not place every tensor on "
            "this hardware -- it likely needs a hand-written mapping."
        ),
    )


def infeasibility_report(
    model: SolverModel, ledger: ResourceLedger, status: str, group: str | None = None
) -> InfeasibilityReportIR:
    """Turn an infeasible solve into a per-resource diagnosis. Uses the solver's IIS (Gurobi) to get the minimal
    conflicting constraint set, maps each back to its resource and limit through its tag, and groups the result so
    a consumer can highlight the offending cores and links with a reason."""
    stats = model.solve_stats()
    iis_available = model.supports_iis
    iis_names: list[str] = []
    if iis_available:
        try:
            model.compute_iis()
            iis_names = model.iis_constraints()
        except Exception as exc:  # noqa: BLE001 -- the IIS is optional evidence on a solve that already failed
            _logger.warning(f"IIS computation failed: {exc}")
            iis_available = False

    grouped: dict[str, dict[str, Any]] = {}
    unbound: list[str] = []
    for name in iis_names:
        tag = ledger.tags.get(name)
        if tag is None or tag.resource is None:
            unbound.append(name)
            continue
        ref = resource_ref(tag.resource)
        entry = grouped.setdefault(f"{ref.kind}:{ref.id}", {"ref": ref, "kinds": set(), "constraints": []})
        if tag.kind is not None:
            entry["kinds"].add(tag.kind)
        entry["constraints"].append(name)

    resources = [_implicated_resource(ledger, entry) for entry in grouped.values()]
    if not resources:
        resources = _direct_capacity_overflows(ledger)

    structural_names = list(unbound) + [c for r in resources if r.unmet is None for c in r.constraints]
    conflicts = _structural_conflicts(ledger, structural_names)

    quantified = next((r for r in resources if r.unmet is not None), None)
    involved_cores = ", ".join(r.resource.label for r in resources if r.resource.kind == "core")
    if quantified is not None and quantified.unmet is not None:
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


def _structural_conflicts(ledger: ResourceLedger, names: list[str]) -> list[StructuralConflictIR]:
    """Group the constraints that enforce a structural rule into plain-language conflicts; the rest are left out."""
    buckets: dict[StructuralRule, StructuralConflictIR] = {}
    for name in names:
        tag = ledger.tags.get(name)
        if tag is None or tag.rule is None:
            continue
        bucket = buckets.setdefault(
            tag.rule, StructuralConflictIR(title=tag.rule.title, explanation=tag.rule.explanation, constraints=[])
        )
        bucket.constraints.append(name)
    return list(buckets.values())


def resource_ref(resource: Resource) -> ResourceRefIR:
    """A physical resource as the IR the architecture view highlights."""
    if isinstance(resource, Core):
        detail = {"core_type": str(resource.core_type)}
        with suppress(AssertionError):
            detail["memory_capacity_bits"] = str(resource.get_memory_capacity())
        return ResourceRefIR(kind="core", id=str(resource.id), label=f"Core {resource.id}", detail=detail)
    return ResourceRefIR(kind="link", id=resource_key(resource), label=resource_key(resource))


def _implicated_resource(ledger: ResourceLedger, entry: dict[str, Any]) -> ImplicatedResourceIR:
    """The diagnosis of one implicated resource: its reason and, when a quantified limit among its conflicts is
    over its bound, that limit's unmet inequality."""
    kinds: list[ResourceKind] = sorted(entry["kinds"], key=lambda kind: kind.name)
    ref = entry["ref"]
    unmet: UnmetConstraintIR | None = None
    if ref.kind == "core" and ref.id.isdigit():
        for kind in kinds:
            unmet = _unmet(ledger, kind, int(ref.id), _forced_terms(ledger, kind, int(ref.id), entry["constraints"]))
            if unmet is not None:
                break
    if unmet is not None and unmet.gap > 0:
        reason = "; ".join(kind.reason for kind in kinds)
    else:
        unmet = None
        if kinds and all(kind.unit for kind in kinds):
            reason = "involved in a reuse/scheduling conflict (not over budget)"
        elif kinds:
            reason = "; ".join(kind.reason for kind in kinds)
        else:
            reason = "conflicting allocation constraints"
    return ImplicatedResourceIR(
        resource=ref,
        constraint_kinds=[kind.name for kind in kinds],
        reason=reason,
        constraints=entry["constraints"],
        unmet=unmet,
    )


def _forced_terms(ledger: ResourceLedger, kind: ResourceKind, core_id: int, names: Iterable[str]) -> dict[str, Any]:
    """The demand terms the IIS actually forces onto this core: the tensors whose demand on it, under this limit or
    under none in particular, one of its constraints carries."""
    terms = ledger.terms.get((kind, core_id), {})
    forced: dict[str, Any] = {}
    for name in names:
        tag = ledger.tags.get(name)
        if tag is None or tag.subject not in terms or tag.kind not in (None, kind):
            continue
        if isinstance(tag.resource, Core) and tag.resource.id == core_id:
            forced[tag.subject] = terms[tag.subject]
    return forced


def _unmet(
    ledger: ResourceLedger, kind: ResourceKind, core_id: int, forced: dict[str, Any]
) -> UnmetConstraintIR | None:
    """The ``demand <= bound`` inequality of a limit on one core from the selected demand terms, with the input
    each side comes from and the levers that would make it fit; None for an unquantified limit."""
    bound = ledger.bounds.get((kind, core_id))
    if not kind.unit or bound is None or not forced:
        return None
    demand = sum(_term_value(v) for v in forced.values())
    gap = demand - bound
    core = f"Core {core_id}"
    terms = []
    for label, info in sorted(forced.items(), key=lambda kv: -_term_value(kv[1])):
        dims, dtype = _term_meta(info)
        terms.append(
            ConstraintTermIR(
                label=label,
                value=_term_value(info),
                detail=kind.term_detail,
                dtype=dtype,
                dims=[TileDimIR(label=lbl, size=sz) for lbl, sz in dims],
            )
        )
    return UnmetConstraintIR(
        family=kind.name,
        statement=(
            f"{kind.demand_label} {core} need {_fmt_qty(demand, kind.unit)}, but the "
            f"{kind.bound_label} {core} is {_fmt_qty(bound, kind.unit)} (short by {_fmt_qty(gap, kind.unit)})"
        ),
        demand_label=f"{kind.demand_label} {core}",
        demand_value=demand,
        demand_input=kind.demand_input,
        bound_label=f"{kind.bound_label} {core}",
        bound_value=bound,
        bound_input=kind.bound_input,
        operator="<=",
        gap=gap,
        unit=kind.unit,
        terms=terms,
        levers=[lever.format(core=core) for lever in kind.levers],
    )


def _direct_capacity_overflows(ledger: ResourceLedger) -> list[ImplicatedResourceIR]:
    """Backend-agnostic diagnosis for a solver without an IIS (OR-Tools): each limit whose recorded demand on a core
    exceeds its non-zero bound, most over first, as the same unmet inequality the IIS path produces."""
    cores_by_id = {tag.resource.id: tag.resource for tag in ledger.tags.values() if isinstance(tag.resource, Core)}
    overflows: list[tuple[float, str, ResourceKind, int]] = []
    for (kind, core_id), terms in ledger.terms.items():
        bound = ledger.bounds.get((kind, core_id))
        if not bound or bound <= 0:
            continue
        demand = sum(_term_value(v) for v in terms.values())
        if demand > bound:
            overflows.append((demand - bound, kind.name, kind, core_id))
    overflows.sort(key=lambda o: (o[0], o[1], o[3]), reverse=True)
    resources: list[ImplicatedResourceIR] = []
    for _gap, _name, kind, core_id in overflows:
        core = cores_by_id.get(core_id)
        ref = (
            resource_ref(core)
            if core is not None
            else ResourceRefIR(kind="core", id=str(core_id), label=f"Core {core_id}")
        )
        resources.append(
            ImplicatedResourceIR(
                resource=ref,
                constraint_kinds=[kind.name],
                reason=kind.reason,
                constraints=[],
                unmet=_unmet(ledger, kind, core_id, dict(ledger.terms[(kind, core_id)])),
            )
        )
    return resources


def _fmt_qty(value: float, unit: str) -> str:
    """A quantity for the diagnosis: bytes scale to KB or MB, counts stay integral."""
    if unit != "bytes":
        return f"{int(round(value))} {unit}"
    if abs(value) >= 1 << 20:
        return f"{value / (1 << 20):.2f} MB"
    if abs(value) >= 1 << 10:
        return f"{value / (1 << 10):.1f} KB"
    return f"{value:.0f} B"


def _term_value(info: Any) -> float:
    """A demand term is either a bare value or a ``{value, dims, dtype}`` record."""
    return float(info["value"]) if isinstance(info, dict) else float(info)


def _term_meta(info: Any) -> tuple[list[tuple[str, int]], str]:
    """The tile shape (per-dim label and size) and dtype of a tensor demand term, if it carries them."""
    if isinstance(info, dict):
        return info.get("dims", []), info.get("dtype", "")
    return [], ""
