"""Infeasibility diagnosis IR: why a mapping did not fit, as an inspectable result (the solver IIS
mapped back to the physical cores/links it over-constrains) rather than a crash."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResourceRefIR(BaseModel):
    """A physical hardware resource a constraint binds -- the thing to highlight on the architecture view."""

    kind: str = Field(description="Physical resource kind, e.g. 'core' or 'link'")
    id: str = Field(description="Stable id matching the accelerator IR (e.g. a core id as a string)")
    label: str = Field(description="Human label, e.g. 'Core 0'")
    detail: dict[str, str] = Field(default_factory=dict, description="Tooltip facts, e.g. capacity vs. required")


class TileDimIR(BaseModel):
    """One dimension of a tensor tile: its loop-dim label and the tile size along it."""

    label: str = Field(description="Loop-dim symbol, e.g. 'z32'")
    size: int = Field(description="Tile size along this dimension")


class ConstraintTermIR(BaseModel):
    """One additive contributor to a resource's demand (e.g. a resident tensor), with its value and,
    for a tensor, its per-dimension tile shape and dtype."""

    label: str = Field(description="What the term is, e.g. a tensor name")
    value: float = Field(description="Its contribution in the unmet-constraint's unit")
    detail: str = Field(default="", description="Extra context, e.g. 'min 1 tile'")
    dtype: str = Field(default="", description="Element type of a tensor term, e.g. 'bf16'")
    dims: list[TileDimIR] = Field(
        default_factory=list, description="Per-dimension tile sizes whose product (× dtype) makes up the value"
    )


class UnmetConstraintIR(BaseModel):
    """The derived hardware constraint that cannot be met, as an intuitive inequality traced to the
    workload/mapping demand and the hardware bound, with ``levers`` that would make it fit."""

    family: str = Field(description="Constraint family, e.g. 'memory_capacity'")
    statement: str = Field(description="One-line intuitive inequality with numbers")
    demand_label: str = Field(description="What is demanded, e.g. 'tensors resident on Core 3'")
    demand_value: float
    demand_input: str = Field(description="Input(s) that set the demand, e.g. 'workload tensor sizes x mapping tiling'")
    bound_label: str = Field(description="The hardware limit, e.g. 'Core 3 on-chip memory'")
    bound_value: float
    bound_input: str = Field(description="Input that sets the bound, e.g. 'hardware: Core 3 memory size'")
    operator: str = Field(default="<=", description="The relation the demand must satisfy")
    gap: float = Field(description="demand - bound in the unit; > 0 is the amount by which it overflows")
    unit: str = Field(description="Unit of the values, e.g. 'bytes'")
    terms: list[ConstraintTermIR] = Field(default_factory=list, description="Breakdown of the demand")
    levers: list[str] = Field(default_factory=list, description="Concrete input changes that would satisfy it")


class ImplicatedResourceIR(BaseModel):
    """One physical resource the infeasibility is pinned to, with the reason, IIS constraint names,
    and the unmet inequality when quantifiable."""

    resource: ResourceRefIR
    constraint_kinds: list[str] = Field(description="Constraint families over-constraining this resource")
    reason: str = Field(description="Human-readable cause, e.g. 'on-chip memory capacity exceeded'")
    constraints: list[str] = Field(description="IIS constraint names bound to this resource")
    unmet: UnmetConstraintIR | None = Field(
        default=None, description="The quantitative unmet constraint (bound vs demand + levers), when derivable"
    )


class InfeasibilityReportIR(BaseModel):
    """The minimal conflict (IIS) mapped back to physical resources -- an inspectable result produced
    instead of crashing on an infeasible mapping."""

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
    """Raised when the MILP has no solution; carries the structured :class:`InfeasibilityReportIR`."""

    def __init__(self, report: InfeasibilityReportIR) -> None:
        self.report = report
        super().__init__(report.summary)
