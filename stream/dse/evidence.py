"""The typed reading of one solved run: what an operator's precondition is allowed to look at.

Everything here comes from artifacts the engine already writes — the per-node cost evidence in
``progress.json`` (the DSE inspection contract), the solved :class:`~stream.ir.allocation.AllocationIR`
in ``allocation.json``, and the :class:`~stream.ir.infeasibility.InfeasibilityReportIR` in
``infeasibility.json``. Nothing is re-derived from a schedule trace, and nothing is invented.

THE ONE RULE THAT MATTERS
-------------------------
``evidence: "none"`` is not ``stall_cycles == 0``. A node whose ZigZag estimate fell back to the
scalar cost has no CME, so *nothing about its memory behaviour was modelled*. Reading that absence
as "this level does not stall" would turn a modelling gap into a proof that a mutation is legal,
which is exactly backwards. :attr:`NodeEvidence.memory_levels` is ``None`` in that case and every
query that could be mistaken for a measurement returns ``None`` rather than ``0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Operand namespaces ──────────────────────────────────────────────────────────────────────────
# A memory level declares the *memory* operands it holds (I1/I2/O); ZigZag's capacity-utilization
# vectors are keyed by *layer* operand (A/B/O for a GEMM, I/W/O for a conv) and indexed by that
# operand's own active levels. Joining them needs the correspondence below AND a length check --
# see `NodeEvidence.capacity_utilization`, which refuses the join rather than guessing.
LAYER_TO_MEMORY_OPERAND: dict[str, str] = {"A": "I1", "I": "I1", "B": "I2", "W": "I2", "O": "O"}

CAPACITY_SATURATION = 0.999
"""At or above this, a level is full for that operand: growing it can change the temporal mapping."""

DEFAULT_NOISE_FLOOR_RELATIVE = 0.02
"""Fallback noise floor when the solver reports no optimality gap, as a fraction of one iteration.

Gurobi defines no single MIP gap for the lexicographic multi-objective model this scheduler builds,
so ``SolveStatsIR.mip_gap`` is ``None`` on the default path. None means UNKNOWN, and the honest
response to an unknown floor is a conservative one: 2% of the per-iteration latency is one to two
orders of magnitude above any plausible branch-and-bound gap, so an operator gated on it must clear
a bar it may not strictly have had to. Erring the other way -- reading None as 0 -- would let every
rounding difference count as evidence. `NoiseFloor.known` records which of the two produced it.
"""


# ── Per-node evidence ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryLevelEvidence:
    """One memory level of one node's cost model, as ZigZag measured it.

    ``stall_cycles`` and ``slack_cycles`` are whole-node figures: ZigZag's
    ``stall_or_slack = (real_cycle - allowed_cycle) x (period_count - 1)`` already carries the
    iteration count, so a consumer must never multiply by iterations again.

    ``real_cycle`` is derived from the port's bandwidth, which makes the whole vector a *bandwidth*
    quantity by construction. It vetoes both capacity and bandwidth growth (absent a stall, neither
    changes latency here) but it can only *select* bandwidth growth -- capacity acts indirectly, via
    which temporal mapping LOMA picks, and needs its own signal.
    """

    name: str
    index: int
    operands: tuple[str, ...]
    stall_cycles: float
    slack_cycles: float
    per_port: dict[str, float]
    utilization: float | None
    """Busiest port's occupancy over the node's computation span; > 1 means oversubscribed."""

    @property
    def bandwidth_headroom_cycles(self) -> float:
        """Cycles that widening this level's binding port can actually buy.

        ZigZag takes ``stall_slack_comb = max(...)`` over ports, so relieving the largest stall
        exposes the SECOND largest -- the saving is the gap between them, not the whole value.
        Zero when only one port stalls is wrong too: then the gap runs down to the largest
        non-stalling entry, which is <= 0, so the whole stall is reclaimable.
        """
        if self.stall_cycles <= 0:
            return 0.0
        stalls = sorted(self.per_port.values(), reverse=True)
        runner_up = stalls[1] if len(stalls) > 1 else 0.0
        return max(0.0, stalls[0] - max(runner_up, 0.0))


@dataclass(frozen=True)
class NodeEvidence:
    """One workload node's per-core cost evidence."""

    name: str
    op: str | None
    core_ids: tuple[int, ...]
    evidence: Literal["cme", "none"]
    """"cme" = ZigZag modelled this node's memory hierarchy; "none" = it did not, so nothing below
    is a measurement. Never conflate the two."""
    latency_cycles: float | None
    ideal_cycles: float | None
    compute_efficiency: float | None
    """ideal / latency. < 1 means the node is not running at its compute-ideal -- the discrepancy
    Rule 2 requires before a core-level operator may be considered for it."""
    mac_spatial_utilization: float | None
    fallback: bool
    memory_levels: dict[str, MemoryLevelEvidence] | None
    """None means NO EVIDENCE, not "no level stalls"."""
    capacity_utilization_raw: dict[str, list[float]] = field(default_factory=dict)
    """ZigZag's ``mem_utili_shared``, keyed by layer operand, on its own per-operand index basis."""

    @property
    def modelled(self) -> bool:
        return self.evidence == "cme" and self.memory_levels is not None

    def stalling_levels(self) -> list[MemoryLevelEvidence]:
        """Levels with a positive stall, largest first. Empty for an unmodelled node -- callers must
        check :attr:`modelled` first; "no stalls found" and "nothing was modelled" are not the same."""
        if not self.modelled:
            return []
        assert self.memory_levels is not None
        stalling = (lv for lv in self.memory_levels.values() if lv.stall_cycles > 0)
        return sorted(stalling, key=lambda lv: -lv.stall_cycles)

    def capacity_utilization(self, level: MemoryLevelEvidence) -> float | None:
        """Fullness of `level` as ZigZag's shared-capacity vector reports it, or None when the two
        index bases cannot be joined.

        The vectors are per layer operand and indexed by that operand's own active levels; the
        levels are per memory operand and indexed by the flat hierarchy. The join is legitimate --
        both orders come from the same ascending ``mem_level_list`` -- but only if the mapped memory
        operand's level count matches the reported vector length. When it does not, the mapping is
        not the one this workload used and the honest answer is None, not a guess.
        """
        if not self.modelled:
            return None
        assert self.memory_levels is not None
        best: float | None = None
        for layer_operand, values in self.capacity_utilization_raw.items():
            memory_operand = LAYER_TO_MEMORY_OPERAND.get(layer_operand.upper())
            if memory_operand is None:
                continue
            active = sorted(
                (lv for lv in self.memory_levels.values() if memory_operand in lv.operands), key=lambda lv: lv.index
            )
            if len(active) != len(values):
                continue  # not this operand's hierarchy: refuse the join rather than mislabel a level
            for position, active_level in enumerate(active):
                if active_level.index == level.index:
                    best = values[position] if best is None else max(best, values[position])
        return best

    def saturated_levels(self) -> list[MemoryLevelEvidence]:
        """Levels that are full for at least one operand -- the capacity signal Rule 1 needs to
        *select* a capacity growth, since the stall vector can only veto it."""
        if not self.modelled:
            return []
        assert self.memory_levels is not None
        out = []
        for level in self.memory_levels.values():
            utilization = self.capacity_utilization(level)
            if utilization is not None and utilization >= CAPACITY_SATURATION:
                out.append(level)
        return sorted(out, key=lambda lv: lv.index)


# ── Whole-run evidence ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResourceSlack:
    resource: str
    kind: str
    slack_cycles: float


@dataclass(frozen=True)
class NoiseFloor:
    """The smallest latency delta this run can distinguish from solver/model noise."""

    cycles: float
    relative: float
    known: bool
    """False when the solver reported no gap and the conservative fallback was used instead."""
    source: str

    def clears(self, predicted_cycles: float) -> bool:
        return predicted_cycles > self.cycles


@dataclass(frozen=True)
class UnmetCapacity:
    """One ``memory_capacity`` conflict from the infeasibility diagnosis: the only signal that names
    both a resource and the amount by which its capacity falls short."""

    resource_kind: str
    resource_id: str
    resource_label: str
    demand_value: float
    bound_value: float
    gap: float
    unit: str
    levers: tuple[str, ...]


@dataclass(frozen=True)
class RunEvidence:
    """Everything one run measured, in the form the operator preconditions read."""

    nodes: tuple[NodeEvidence, ...] = ()
    latency_total: float | None = None
    latency_per_iteration: float | None = None
    overlap_cycles: float | None = None
    compute_bound_cycles: float | None = None
    transfer_bound_cycles: float | None = None
    binding_resources: tuple[str, ...] = ()
    per_resource_slack: tuple[ResourceSlack, ...] = ()
    recurrence_bound_cycles: float = 0.0
    compute_cores_used: int | None = None
    compute_cores_available: int | None = None
    end_to_end_mac_utilization: float | None = None
    latency_weighted_mac_spatial_utilization: float | None = None
    degenerate: bool = False
    solver_status: str | None = None
    mip_gap: float | None = None
    unmet_capacity: tuple[UnmetCapacity, ...] = ()
    feasible: bool = True
    fused_group_layers: tuple[tuple[str, ...], ...] = ()
    """Layers per fused group, in solve order -- what a fusion-cut operator has to name."""

    @property
    def n_fused_groups(self) -> int:
        return len(self.fused_group_layers)

    @property
    def n_nodes(self) -> int:
        return sum(len(g) for g in self.fused_group_layers)

    # ── Derived quantities ──────────────────────────────────────────────────────────────────

    @property
    def initiation_interval(self) -> float | None:
        """II = latency_per_iteration - overlap: the rate the steady state actually sustains.

        Its floor is ``max(RecMII, resource-bound)``; the distance between II and that floor is
        what a system-tier operator can hope to reclaim.
        """
        if self.latency_per_iteration is None:
            return None
        return self.latency_per_iteration - (self.overlap_cycles or 0.0)

    @property
    def noise_floor(self) -> NoiseFloor:
        reference = self.latency_per_iteration or self.latency_total or 0.0
        if self.mip_gap is not None:
            return NoiseFloor(
                cycles=abs(self.mip_gap) * reference,
                relative=abs(self.mip_gap),
                known=True,
                source=f"solver optimality gap ({self.mip_gap:.3g})",
            )
        return NoiseFloor(
            cycles=DEFAULT_NOISE_FLOOR_RELATIVE * reference,
            relative=DEFAULT_NOISE_FLOOR_RELATIVE,
            known=False,
            source=(
                "solver reported no optimality gap (lexicographic multi-objective); "
                f"conservative fallback of {DEFAULT_NOISE_FLOOR_RELATIVE:.0%} of one iteration"
            ),
        )

    def node(self, name: str) -> NodeEvidence | None:
        return next((n for n in self.nodes if n.name == name), None)

    def slack_of(self, resource: str) -> float | None:
        return next((s.slack_cycles for s in self.per_resource_slack if s.resource == resource), None)

    def binding_links(self) -> list[ResourceSlack]:
        """Binding resources that are links, i.e. the ones a bandwidth growth could ever address."""
        binding = set(self.binding_resources)
        return [s for s in self.per_resource_slack if s.kind == "link" and s.resource in binding]

    # ── Construction ────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(
        cls,
        progress: dict[str, Any] | None = None,
        allocation: dict[str, Any] | None = None,
        infeasibility: dict[str, Any] | None = None,
    ) -> RunEvidence:
        """Read the three artifacts a run writes. Any of them may be absent."""
        nodes = _parse_nodes(progress)
        fields: dict[str, Any] = {"nodes": tuple(nodes)}
        fields.update(_parse_allocation(allocation))
        if infeasibility:
            fields["feasible"] = False
            fields["solver_status"] = infeasibility.get("status") or fields.get("solver_status")
            fields["unmet_capacity"] = _parse_unmet_capacity(infeasibility)
        return cls(**fields)


# ── Parsing ─────────────────────────────────────────────────────────────────────────────────────


def _parse_nodes(progress: dict[str, Any] | None) -> list[NodeEvidence]:
    """Per-node cost rows out of the tracker's `core_cost` stage (one entry per fused group)."""
    if not progress:
        return []
    stage = next((s for s in progress.get("stages") or [] if s.get("key") == "core_cost"), None)
    artifact = (stage or {}).get("artifact") or {}
    out: list[NodeEvidence] = []
    for group in artifact.get("groups") or []:
        for row in group.get("nodes") or []:
            out.append(_parse_node(row))
    return out


def _parse_node(row: dict[str, Any]) -> NodeEvidence:
    raw_levels = row.get("memory_levels")
    levels: dict[str, MemoryLevelEvidence] | None = None
    if isinstance(raw_levels, dict):
        levels = {
            name: MemoryLevelEvidence(
                name=name,
                index=int(entry.get("index", 0)),
                operands=tuple(str(o) for o in entry.get("operands") or ()),
                stall_cycles=float(entry.get("stall_cycles") or 0.0),
                slack_cycles=float(entry.get("slack_cycles") or 0.0),
                per_port={str(p): float(v) for p, v in (entry.get("per_port") or {}).items()},
                utilization=_optional_float(entry.get("utilization")),
            )
            for name, entry in raw_levels.items()
        }
    utilization = row.get("operand_memory_utilization") or {}
    return NodeEvidence(
        name=str(row.get("name")),
        op=row.get("op"),
        core_ids=tuple(int(c) for c in row.get("cores") or ()),
        # An explicit "none" and a missing key are the same claim: nothing was modelled.
        evidence="cme" if row.get("evidence") == "cme" else "none",
        latency_cycles=_optional_float(row.get("latency")),
        ideal_cycles=_optional_float(row.get("ideal_cycle")),
        compute_efficiency=_optional_float(row.get("efficiency")),
        mac_spatial_utilization=_optional_float(row.get("mac_utilization")),
        fallback=bool(row.get("estimator") == "ideal-cycle"),
        memory_levels=levels,
        capacity_utilization_raw={
            str(op): [float(v) for v in values] for op, values in (utilization.get("shared") or {}).items()
        },
    )


def _parse_allocation(allocation: dict[str, Any] | None) -> dict[str, Any]:
    """Whole-run quantities from `allocation.json`.

    Multi-group runs carry one AllocationIR per fused group; the group whose latency dominates is
    the one whose pipelining sets the runtime, so it is the one whose overlap evidence is read.
    Averaging binding sets across groups would name a resource that binds nothing.
    """
    if not allocation:
        return {}
    groups = allocation.get("groups") or []
    allocations = [g.get("allocation") for g in groups if g.get("allocation")]
    if not allocations:
        return {}
    dominant = max(allocations, key=lambda a: float((a.get("latency") or {}).get("total") or 0.0))
    performance = dominant.get("performance") or {}
    latency = performance.get("latency") or dominant.get("latency") or {}
    bottleneck = performance.get("bottleneck") or {}
    aggregate = performance.get("aggregate") or {}
    overlap = performance.get("overlap") or {}
    solve = dominant.get("solve") or {}
    return {
        "latency_total": _optional_float(allocation.get("total_latency")) or _optional_float(latency.get("total")),
        "latency_per_iteration": _optional_float(latency.get("per_iteration")),
        "overlap_cycles": _optional_float(latency.get("overlap_between_iterations")),
        "compute_bound_cycles": _optional_float(bottleneck.get("compute_bound_cycles")),
        "transfer_bound_cycles": _optional_float(bottleneck.get("transfer_bound_cycles")),
        "binding_resources": tuple(str(r) for r in overlap.get("binding_resources") or ()),
        "per_resource_slack": tuple(
            ResourceSlack(
                resource=str(s.get("resource")),
                kind=str(s.get("kind")),
                slack_cycles=float(s.get("slack_cycles") or 0.0),
            )
            for s in overlap.get("per_resource_slack") or ()
        ),
        "recurrence_bound_cycles": float(overlap.get("recurrence_bound_cycles") or 0.0),
        "compute_cores_used": _optional_int(aggregate.get("compute_cores_used")),
        "compute_cores_available": _optional_int(aggregate.get("compute_cores_available")),
        "end_to_end_mac_utilization": _optional_float(aggregate.get("end_to_end_mac_utilization")),
        "latency_weighted_mac_spatial_utilization": _optional_float(
            aggregate.get("latency_weighted_mac_spatial_utilization")
        ),
        "degenerate": bool(aggregate.get("degenerate")),
        "solver_status": solve.get("status"),
        # None here means the floor is unknown, which `noise_floor` handles explicitly.
        "mip_gap": _optional_float(solve.get("mip_gap")),
        "fused_group_layers": tuple(
            tuple(str(layer) for layer in fg.get("layers") or ())
            for a in allocations
            for fg in a.get("fused_groups") or ()
        ),
    }


def _parse_unmet_capacity(infeasibility: dict[str, Any]) -> tuple[UnmetCapacity, ...]:
    out: list[UnmetCapacity] = []
    for implicated in infeasibility.get("resources") or []:
        unmet = implicated.get("unmet")
        if not unmet or unmet.get("family") != "memory_capacity":
            continue
        resource = implicated.get("resource") or {}
        out.append(
            UnmetCapacity(
                resource_kind=str(resource.get("kind", "")),
                resource_id=str(resource.get("id", "")),
                resource_label=str(resource.get("label", "")),
                demand_value=float(unmet.get("demand_value") or 0.0),
                bound_value=float(unmet.get("bound_value") or 0.0),
                gap=float(unmet.get("gap") or 0.0),
                unit=str(unmet.get("unit", "")),
                levers=tuple(str(x) for x in unmet.get("levers") or ()),
            )
        )
    return tuple(out)


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
