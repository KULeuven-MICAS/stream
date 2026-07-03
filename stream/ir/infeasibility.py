"""Infeasibility diagnosis IR -- why a mapping did not fit, as an inspectable result rather than a crash.

When the tensor/transfer allocation MILP has no solution, a solver that supports an IIS (Irreducible
Inconsistent Subsystem -- Gurobi today) yields the *minimal* set of mutually-conflicting constraints.
Each resource-bound constraint the allocator adds is tagged with the physical resource it binds (a
core's memory, a link, ...), so the IIS maps straight back to the hardware: the report says which
cores/links are over-constrained and why. This drives a "highlight the offending resources on the
architecture view" visualization, and it is deliberately backend-agnostic and resource-kind-agnostic
so new solvers or new hardware constraints extend it without touching consumers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResourceRefIR(BaseModel):
    """A physical hardware resource a constraint binds -- the thing to highlight on the architecture
    view. ``kind`` is open (``core``/``link``/future memory-bank/DMA-engine) so new resource types
    need no schema change; ``id`` matches the accelerator IR so a consumer can find the rendered node.
    """

    kind: str = Field(description="Physical resource kind, e.g. 'core' or 'link'")
    id: str = Field(description="Stable id matching the accelerator IR (e.g. a core id as a string)")
    label: str = Field(description="Human label, e.g. 'Core 0'")
    detail: dict[str, str] = Field(default_factory=dict, description="Tooltip facts, e.g. capacity vs. required")


class ImplicatedResourceIR(BaseModel):
    """One physical resource the infeasibility is pinned to, with the human reason and the raw IIS
    constraint names that bound it."""

    resource: ResourceRefIR
    constraint_kinds: list[str] = Field(description="Constraint families over-constraining this resource")
    reason: str = Field(description="Human-readable cause, e.g. 'on-chip memory capacity exceeded'")
    constraints: list[str] = Field(description="IIS constraint names bound to this resource")


class InfeasibilityReportIR(BaseModel):
    """The minimal conflict (IIS) mapped back to physical resources -- produced instead of crashing so
    a launch with an infeasible mapping still yields an inspectable result. ``resources`` drives the
    architecture-view highlight; ``unbound_constraints`` are IIS members with no single physical
    resource (structural couplings), reported textually.
    """

    model_config = ConfigDict(
        json_schema_extra={"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "stream/infeasibility/v1"}
    )

    schema_version: Literal["1.0"] = "1.0"
    feasible: Literal[False] = False
    status: str = Field(description="Solver status, e.g. 'INFEASIBLE' or 'TIME_LIMIT'")
    backend: str = Field(description="Solver backend, e.g. 'GUROBI'")
    solver: str = Field(description="Underlying solver, e.g. 'gurobi'")
    group: str | None = Field(default=None, description="Fusion group this diagnosis is for, if per-group")
    iis_available: bool = Field(description="False when the backend cannot compute an IIS (e.g. OR-Tools)")
    resources: list[ImplicatedResourceIR] = Field(default_factory=list)
    unbound_constraints: list[str] = Field(
        default_factory=list, description="IIS constraints not bound to one physical resource (structural)"
    )
    summary: str = Field(description="One-line human summary of why it did not fit")


class InfeasibleAllocationError(RuntimeError):
    """Raised by the allocator when the MILP has no solution. Carries the structured
    :class:`InfeasibilityReportIR` so a caller can surface a diagnosis instead of a bare failure."""

    def __init__(self, report: InfeasibilityReportIR) -> None:
        self.report = report
        super().__init__(report.summary)
