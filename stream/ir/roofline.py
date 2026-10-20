"""Typed roofline of an accelerator and of the solved groups of a run (see stream.cost_model.roofline)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RoofIR(BaseModel):
    """The compute roof of one class of compute core."""

    name: str = Field(description="'MAC' or 'vector'")
    ops_per_cycle: float
    cores: list[int]


class CeilingResourceIR(BaseModel):
    """One memory port or link a bandwidth ceiling sums."""

    resource: str = Field(
        description="'<memory>.<port>@<cores>', 'measured@<core>' for a measured shared bandwidth, "
        "'<sender>-><receiver>' or 'bus <cores>'"
    )
    cores: list[int]
    bytes_per_cycle: float


class CeilingIR(BaseModel):
    """The bandwidth ceiling of one level of the memory hierarchy."""

    name: str = Field(description="'off-chip memory', 'off-chip links', 'on-chip memory' or 'core-to-core links'")
    bytes_per_cycle: float
    resources: list[CeilingResourceIR]


class HardwareRooflineIR(BaseModel):
    """An accelerator's roofline: its compute roofs and the bandwidth ceiling of each memory level."""

    roofs: list[RoofIR] = Field(description="Largest first")
    levels: list[CeilingIR]


class PointLevelIR(BaseModel):
    """A group's traffic through one level and the ceiling it gives the group."""

    name: str
    bytes: float = Field(description="Bytes the group moved through the level over its iterations")
    intensity: float | None = Field(description="Operations per byte moved")
    attainable: float | None = Field(description="The level's bandwidth times the intensity, in operations per cycle")
    busiest: dict[str, Any] | None = Field(
        description="The level's busiest resource: 'resource', its name, and 'utilization', its real cycles per "
        "initiation interval"
    )


class RooflinePointIR(BaseModel):
    """One solved group on the roofline."""

    layers: list[str]
    operators: list[str]
    roof: str | None = Field(description="The roof it is read against")
    macs: float
    operations: float = Field(description="Its MACs, or its elementwise operations without MACs")
    latency_cycles: int
    fill_cycles: int
    iterations: int
    ops_per_cycle: float = Field(description="What it achieves")
    levels: list[PointLevelIR]
    bound: str | None = Field(description="The lowest ceiling: 'compute' or a level")
    attainable: float | None = Field(description="That ceiling, in operations per cycle")
    efficiency: float | None = Field(description="ops_per_cycle / attainable")
    spatial_utilization: float | None
    limiter: str | None = Field(description="The lowest ceiling counting the fill of the MAC array ('array fill')")
    limit: float | None
    diagnosis: list[dict[str, Any]] = Field(
        description="What holds it down, the lowest ceiling first, each a 'code' and its details: 'compute_bound', "
        "'memory_bound' ('level'), 'array_underfilled' ('spatial_utilization'), 'exceeds_ceiling' and "
        "'resource_saturated' ('level' and its busiest 'resource' and 'utilization'), 'below_ceiling' ('fraction' of "
        "the limit achieved, 'fill_share' of the latency spent filling), or 'no_mac_work' alone"
    )
    signature: str | None = Field(default=None, description="Equal for groups that are the same point")
    groups: list[int] = Field(default_factory=list, description="In a run, the groups it stands for")
    multiplicity: int = 1
    share: float | None = Field(default=None, description="In a run, its groups' share of the run's latency")


class RooflineIR(BaseModel):
    """The roofline of a run: the hardware and its solved groups, repeated groups merged into one point."""

    hardware: HardwareRooflineIR
    points: list[RooflinePointIR]
    groups_solved: int
    n_groups: int
    summary: dict[str, Any] = Field(
        description="'latency_cycles', 'macs' and 'operations' of the solved groups, and 'bound_share' and "
        "'limiter_share', the share of that latency per bound and per limiter ('unplotted' for points without one)"
    )
