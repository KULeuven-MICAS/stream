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


class TileDimIR(BaseModel):
    """One dimension of a tensor tile: its loop-dim label (the affine symbol also shown in the graph
    view) and the tile size along it -- so the per-dimension sizes that make up a tensor's tile are
    visible, not just the total."""

    label: str = Field(description="Loop-dim symbol, e.g. 'z32'")
    size: int = Field(description="Tile size along this dimension")


class ConstraintTermIR(BaseModel):
    """One additive contributor to a resource's demand -- e.g. a tensor that must be resident -- with
    the value it adds and, for a tensor, its per-dimension tile shape + dtype, so a designer sees
    exactly which tile (and why it is large) fills the resource up."""

    label: str = Field(description="What the term is, e.g. a tensor name")
    value: float = Field(description="Its contribution in the unmet-constraint's unit")
    detail: str = Field(default="", description="Extra context, e.g. 'min 1 tile'")
    dtype: str = Field(default="", description="Element type of a tensor term, e.g. 'bf16'")
    dims: list[TileDimIR] = Field(
        default_factory=list, description="Per-dimension tile sizes whose product (× dtype) makes up the value"
    )


class UnmetConstraintIR(BaseModel):
    """The derived hardware constraint that cannot be met, stated as an intuitive inequality with real
    numbers and traced to the *inputs* that set each side -- so a designer knows what to change.

    Reads: ``<demand_label> = <demand_value> <unit> <operator> <bound_value> <unit> = <bound_label>`` is
    violated (short by ``gap``). ``demand`` comes from the workload/mapping, ``bound`` from the
    hardware; ``levers`` list the concrete input edits that would make it fit.
    """

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
    """One physical resource the infeasibility is pinned to, with the human reason, the raw IIS
    constraint names that bound it, and -- when quantifiable -- the unmet inequality in designer terms."""

    resource: ResourceRefIR
    constraint_kinds: list[str] = Field(description="Constraint families over-constraining this resource")
    reason: str = Field(description="Human-readable cause, e.g. 'on-chip memory capacity exceeded'")
    constraints: list[str] = Field(description="IIS constraint names bound to this resource")
    unmet: UnmetConstraintIR | None = Field(
        default=None, description="The quantitative unmet constraint (bound vs demand + levers), when derivable"
    )


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
