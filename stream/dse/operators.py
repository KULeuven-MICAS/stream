"""The guarded operator registry: what a DSE agent is *allowed* to change, and why.

An operator is a quadruple:

``precondition``
    A boolean over measured evidence (:mod:`stream.dse.evidence`). **An operator whose
    precondition no evidence satisfies is never offered.** That alone removes the failure mode
    where a search spends its budget changing things that cannot affect the result.
``effect``
    The concrete edit — to a :class:`~stream.hardware.bundle.HardwareBundle` or to the mapping
    parameters. There is exactly one implementation, in :func:`apply_operator`, so what is offered
    and what is applied cannot drift apart.
``couples``
    Invariants that must move with the edit. Resizing a compute array is the sharp case: it never
    emits a bare change, it atomically rescales the register file — per level, and only for the
    core-local ones (see :func:`_resize_array`).
``predicted_delta``
    An expected cycle saving *with its derivation and its units*, always an upper bound. This is
    what makes a proposal falsifiable: the run that follows either recovers it or does not.

THE FOUR RULES
--------------
1. **The veto.** No stall at a memory level ⇒ growing that level's size *or* its bandwidth is
   illegal: absent a stall, neither changes latency there. The stall vector is a bandwidth quantity
   by construction (``real_cycle`` is bandwidth-derived), so it can *select* a bandwidth growth but
   not a capacity one — capacity is selected from the infeasibility report or a saturated
   ``mem_utili_shared``. And ``evidence: "none"`` is never "no stall": a missing CME is a missing
   measurement, and the operator is simply not offered.
2. **Ordering.** Core-tier operators are considered before system-tier ones, and only for nodes
   whose CME shows a discrepancy (``compute_efficiency < 1``).
3. **System tier.** Fusion / intra-core-tile / core-count operators are selected from the solver's
   own binding set and the II decomposition (``II = latency_per_iteration - overlap``, floored by
   ``recurrence_bound_cycles``).
4. **NoC and off-chip.** Growing a link needs *exposed* (non-overlapped) cycles on that specific
   resource and a predicted saving above the noise floor — never on a compute-bound schedule.
   Bandwidth is the knob that always looks helpful and, under the occupancy model, usually is not.

Every hardware edit is priced against a :class:`~stream.hardware.cost.HardwareBudget` and rejected
*before* any solve if it busts it. Without that, "minimise latency" degenerates into "grow
everything", because the engine's own ``unit_area: 0`` makes growth free.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from stream.dse.evidence import MemoryLevelEvidence, NodeEvidence, NoiseFloor, RunEvidence
from stream.hardware.bundle import HardwareBundle
from stream.hardware.cost import HardwareBudget, check_budget

# ── Tunables ────────────────────────────────────────────────────────────────────────────────────

DEFAULT_BUDGET_HEADROOM = 0.10
"""Fraction of the baseline area/power a variant may exceed when no explicit budget is given."""

GROWTH_FACTORS: tuple[int, ...] = (2, 4)
"""Declared range for a single growth step. A step budget, not a free variable: an operator that
could ask for any factor would simply ask for the largest one that still fits the budget."""

BANDWIDTH_STEP = 2
"""One doubling per wave, deliberately.

The predicted saving from widening a port is the gap between the largest and second-largest port
stall, and that gap does not depend on the factor -- a 4x port buys the same ceiling as a 2x port
and costs twice the column IO. So the step is fixed at the smallest one and a further widening has
to be justified by a fresh measurement, not by the same one twice.
"""

ARRAY_SCALE_FACTORS: tuple[int, ...] = (2,)
"""Array resizes are offered one doubling at a time -- the coupling below has to hold for each."""

MAC_ARRAY_SATURATED = 0.999
"""At/above this the spatial mapping already fills the array, so a bigger array does more per cycle."""

LOAD_IMBALANCE = 1.25
"""Ratio of the busiest resource's occupancy to the mean before a rebalance is worth offering."""

COMPUTE_BOUND_PCT = 50.0
"""Above this share of per-iteration cycles the schedule is compute-bound and Rule 4 refuses."""

MIN_FUSIBLE_LAYERS = 2
"""A single-layer workload has no fusion boundary to move."""


class OperatorTier(StrEnum):
    CORE = "core"
    SYSTEM = "system"
    LINK = "link"


OFFER_TIERS: tuple[OperatorTier, ...] = (OperatorTier.CORE, OperatorTier.SYSTEM, OperatorTier.LINK)
"""Rule 2's ordering, made data: offers are returned in this order and never re-sorted by size."""


# ── Declarations ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictedDelta:
    """An expected saving, always an upper bound, always with its units spelled out."""

    cycles: float
    scope: Literal["node", "iteration"]
    """"node": cycles off ONE node's contribution to ONE steady-state iteration. ZigZag's
    ``stall_or_slack`` already carries ``(period_count - 1)``, so this is a whole-node figure and a
    consumer must NOT multiply by an iteration count again. "iteration": cycles off one steady-state
    iteration of the whole schedule."""
    derivation: str
    unit: str = "cycles"
    bound: Literal["upper"] = "upper"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "unit": self.unit,
            "scope": self.scope,
            "bound": self.bound,
            "derivation": self.derivation,
        }


ALIAS_COUPLING = (
    "a level listed in the bundle's memory_aliases is ONE physical memory seen from several cores: "
    "every view is scaled together, or the edit changes what the solve sees without changing what "
    "the cost model bills"
)


@dataclass(frozen=True)
class Operator:
    """What an operator is, independent of any particular run."""

    id: str
    tier: OperatorTier
    kind: Literal["hardware", "mapping"]
    summary: str
    couples: tuple[str, ...] = ()


OPERATORS: dict[str, Operator] = {
    op.id: op
    for op in (
        Operator(
            id="core.memory.bandwidth",
            tier=OperatorTier.CORE,
            kind="hardware",
            summary="Widen the ports of a memory level that demonstrably stalls the array.",
            couples=(ALIAS_COUPLING,),
        ),
        Operator(
            id="core.memory.capacity",
            tier=OperatorTier.CORE,
            kind="hardware",
            summary="Grow a memory level that is full, so the temporal mapping can keep an operand resident.",
            couples=(ALIAS_COUPLING,),
        ),
        Operator(
            id="core.array.resize",
            tier=OperatorTier.CORE,
            kind="hardware",
            summary="Scale a compute array, rescaling the core-local memories that serve the resized dimensions.",
            couples=(
                "capacity AND bandwidth of every core-local level whose served_dimensions "
                "intersect the resized dimensions, by the product of those dimensions' factors",
                "levels shared with another core (memory_aliases) are NEVER scaled: the array is "
                "per core, the shared scratchpad is not",
                "post-hoc: reject the variant if mac_spatial_utilization drops",
            ),
        ),
        Operator(
            id="system.tiling.intra_core",
            tier=OperatorTier.SYSTEM,
            kind="mapping",
            summary="Change the intra-core (layer-fusion) tile extent of a fused group.",
        ),
        Operator(
            id="system.fusion.cut",
            tier=OperatorTier.SYSTEM,
            kind="mapping",
            summary="Move the fusion boundaries: which layers share on-chip residency.",
        ),
        Operator(
            id="system.alloc.cores",
            tier=OperatorTier.SYSTEM,
            kind="mapping",
            summary="Let the mapper spread work over more cores (nb_cols_to_use).",
        ),
        Operator(
            id="link.bandwidth",
            tier=OperatorTier.LINK,
            kind="hardware",
            summary="Widen an interconnect or off-chip link that binds the inter-iteration overlap.",
        ),
    )
}


@dataclass(frozen=True)
class Offer:
    """One legal move on this run's evidence."""

    operator_id: str
    tier: OperatorTier
    kind: Literal["hardware", "mapping"]
    target: dict[str, Any]
    args: dict[str, Any]
    evidence: str
    """The measured facts that satisfied the precondition, in numbers."""
    effect: str
    couples: tuple[str, ...]
    predicted_delta: PredictedDelta
    cost: dict[str, Any] | None = None
    """Area/energy of the resulting bundle against the budget, for a hardware edit."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator_id,
            "tier": str(self.tier),
            "kind": self.kind,
            "target": self.target,
            "args": self.args,
            "evidence": self.evidence,
            "effect": self.effect,
            "couples": list(self.couples),
            "predicted_delta": self.predicted_delta.as_dict(),
            "cost": self.cost,
        }


@dataclass(frozen=True)
class Veto:
    """A move that was considered and refused, with the rule that refused it.

    Vetoes are reported, not silently dropped: "this was never legal" is the most informative
    thing the registry can say, and a reader who cannot see it will keep proposing the move.
    """

    operator_id: str
    target: dict[str, Any]
    rule: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"operator": self.operator_id, "target": self.target, "rule": self.rule, "reason": self.reason}


@dataclass(frozen=True)
class OfferResult:
    offered: list[Offer]
    vetoed: list[Veto]
    noise_floor: NoiseFloor
    budget_label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            # `clears_noise_floor` is reported on every offer, not just the link tier Rule 4 gates on
            # it. A core-level move that removes a real stall worth 463 cycles on a 187,082-cycle
            # iteration is legal and is still invisible end to end; a ranker has to be able to see
            # that, and the guards deliberately do not decide it.
            "offered": [
                {**o.as_dict(), "clears_noise_floor": self.noise_floor.clears(o.predicted_delta.cycles)}
                for o in self.offered
            ],
            "vetoed": [v.as_dict() for v in self.vetoed],
            "noise_floor": {
                "cycles": self.noise_floor.cycles,
                "relative": self.noise_floor.relative,
                "known": self.noise_floor.known,
                "source": self.noise_floor.source,
            },
            "budget": self.budget_label,
        }

    def ids(self) -> list[str]:
        return [o.operator_id for o in self.offered]


@dataclass(frozen=True)
class AppliedOperator:
    """The result of applying an operator: the edited bundle and/or mapping parameters."""

    operator_id: str
    bundle: HardwareBundle | None
    mapping_params: dict[str, Any]
    notes: list[str] = field(default_factory=list)


# ── Offering ────────────────────────────────────────────────────────────────────────────────────


def offer_operators(
    evidence: RunEvidence,
    *,
    bundle: HardwareBundle | None = None,
    budget: HardwareBudget | None = None,
    mapping_params: dict[str, Any] | None = None,
) -> OfferResult:
    """Every legal move on this evidence, in Rule 2's tier order, plus what was refused and why.

    ``bundle`` is the hardware the evidence was measured on. Without it no hardware operator can be
    offered at all: its edit could not be priced, and an unpriced edit cannot be certified within
    budget. That is the same discipline as ``evidence: "none"`` — we do not offer what we cannot
    check.
    """
    mapping_params = dict(mapping_params or {})
    if budget is None and bundle is not None:
        budget = HardwareBudget.from_bundle(bundle, DEFAULT_BUDGET_HEADROOM)
    offers: list[Offer] = []
    vetoes: list[Veto] = []

    for propose in (_core_memory, _core_array, _system, _link):
        found, refused = propose(evidence, bundle, budget, mapping_params)
        offers.extend(found)
        vetoes.extend(refused)

    offers = _dedupe(offers)
    # Rule 2 is an ordering over tiers, not over predicted sizes: a core-level fix that removes a
    # measured stall is considered before a system-level reshuffle even when the latter predicts more.
    offers.sort(key=lambda o: OFFER_TIERS.index(o.tier))
    return OfferResult(
        offered=offers,
        vetoed=vetoes,
        noise_floor=evidence.noise_floor,
        budget_label=budget.label if budget else None,
    )


def _dedupe(offers: list[Offer]) -> list[Offer]:
    """Collapse offers whose EDIT is identical, keeping the one with the largest predicted saving.

    Three GEMM nodes sharing one core set all stall on the same scratchpad, so they all propose the
    same widening. Listing it three times would make one move look like three pieces of evidence.
    """
    best: dict[str, Offer] = {}
    for offer in offers:
        key = f"{offer.operator_id}|{sorted(offer.args.items(), key=str)}"
        current = best.get(key)
        if current is None or offer.predicted_delta.cycles > current.predicted_delta.cycles:
            best[key] = offer
    return list(best.values())


# ── Rule 1 + Rule 2: core memory ────────────────────────────────────────────────────────────────


def _core_memory(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    mapping_params: dict[str, Any],
) -> tuple[list[Offer], list[Veto]]:
    """Rule 1 (the veto) and Rule 2 (only nodes with a discrepancy), for both memory knobs."""
    del mapping_params
    offers: list[Offer] = []
    vetoes: list[Veto] = []

    for node in evidence.nodes:
        target = {"node": node.name, "cores": list(node.core_ids)}
        if node.evidence != "cme" or node.memory_levels is None:
            # THE case the whole rule exists for. No CME means the memory hierarchy was not
            # modelled for this node, so there is no stall vector to read. Treating that as "no
            # stall" would convert a modelling gap into a licence to grow.
            vetoes.append(
                Veto(
                    "core.memory.*",
                    target,
                    "rule-1",
                    f"node '{node.name}' has evidence=none (no CME): its memory behaviour was not "
                    "modelled, so the absence of a stall is not evidence that a growth is legal",
                )
            )
            continue
        if node.compute_efficiency is not None and node.compute_efficiency >= 1.0:
            vetoes.append(
                Veto(
                    "core.memory.*",
                    target,
                    "rule-2",
                    f"node '{node.name}' runs at its compute-ideal (compute_efficiency="
                    f"{node.compute_efficiency:.4g}); there is no discrepancy for a core-level operator to close",
                )
            )
            continue

        saturated = {lv.index for lv in node.saturated_levels()}
        for level in sorted(node.memory_levels.values(), key=lambda lv: lv.index):
            level_target = {**target, "memory": level.name, "level_index": level.index}
            if level.stall_cycles <= 0:
                vetoes.append(
                    Veto(
                        "core.memory.*",
                        level_target,
                        "rule-1",
                        f"'{level.name}' does not stall (slack {level.slack_cycles:.0f} cycles): neither "
                        "more capacity nor more bandwidth can change latency at this level",
                    )
                )
                continue
            offers.extend(_bandwidth_offers(node, level, level_target, bundle, budget, vetoes))
            offers.extend(_capacity_offers(node, level, level_target, bundle, budget, vetoes, level.index in saturated))

    offers.extend(_infeasible_capacity_offers(evidence, bundle, budget, vetoes))
    return offers, vetoes


def _bandwidth_offers(
    node: NodeEvidence,
    level: MemoryLevelEvidence,
    target: dict[str, Any],
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    vetoes: list[Veto],
) -> list[Offer]:
    headroom = level.bandwidth_headroom_cycles
    ports = ", ".join(f"{p}={v:.0f}" for p, v in sorted(level.per_port.items(), key=lambda kv: -kv[1]))
    if headroom <= 0:
        vetoes.append(
            Veto(
                "core.memory.bandwidth",
                target,
                "rule-1",
                f"'{level.name}' stalls {level.stall_cycles:.0f} cycles but two ports are tied at the "
                f"maximum ({ports}); widening the binding one only exposes the other, buying 0 cycles",
            )
        )
        return []
    derivation = (
        f"stall_slack_comb = max over ports, so relieving the largest ({max(level.per_port.values()):.0f}) "
        f"exposes the second largest; the saving is the gap between them. Ports: {ports}. "
        "stall_or_slack already carries (period_count - 1), so this is a whole-node figure for one "
        "steady-state iteration and must not be multiplied by an iteration count."
    )
    args = {"cores": list(node.core_ids), "memory": level.name, "factor": BANDWIDTH_STEP}
    cost = _price("core.memory.bandwidth", args, bundle, budget, target, vetoes)
    if cost is None:
        return []
    occupancy = f"{level.utilization:.3g}" if level.utilization is not None else "unknown"
    return [
        Offer(
            operator_id="core.memory.bandwidth",
            tier=OperatorTier.CORE,
            kind="hardware",
            target=target,
            args=args,
            evidence=(
                f"node '{node.name}' runs at compute_efficiency={node.compute_efficiency:.4g} "
                f"({node.ideal_cycles:.0f} ideal vs {node.latency_cycles:.0f} actual cycles) and "
                f"'{level.name}' stalls {level.stall_cycles:.0f} cycles (port occupancy {occupancy})"
            ),
            effect=(
                f"multiply every port bandwidth of '{level.name}' on cores {list(node.core_ids)} by {BANDWIDTH_STEP}"
            ),
            couples=OPERATORS["core.memory.bandwidth"].couples,
            predicted_delta=PredictedDelta(cycles=headroom, scope="node", derivation=derivation),
            cost=cost,
        )
    ]


def _capacity_offers(  # noqa: PLR0913 -- one call site; the arguments are the guard's inputs
    node: NodeEvidence,
    level: MemoryLevelEvidence,
    target: dict[str, Any],
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    vetoes: list[Veto],
    saturated: bool,
) -> list[Offer]:
    """Capacity growth on a feasible run: the stall vetoes it, saturation selects it.

    The stall alone cannot select capacity. ``stall_or_slack = (real_cycle - allowed_cycle) x
    (period_count - 1)`` with ``real_cycle`` bandwidth-derived, so it is a bandwidth quantity by
    construction; capacity acts only indirectly, by changing which temporal mapping LOMA picks.
    """
    if not saturated:
        vetoes.append(
            Veto(
                "core.memory.capacity",
                target,
                "rule-1",
                f"'{level.name}' stalls, so the veto passes, but nothing selects CAPACITY: its shared "
                "utilization is below saturation and no infeasibility names it. The stall is a "
                "bandwidth quantity by construction and cannot choose between the two knobs",
            )
        )
        return []
    utilization = node.capacity_utilization(level)
    derivation = (
        "capacity does not enter the stall arithmetic at all; it acts by letting LOMA choose a "
        f"temporal mapping that keeps the operand resident. The ceiling is therefore the level's own "
        f"stall ({level.stall_cycles:.0f} cycles, whole-node, one steady-state iteration) and the "
        "realised saving may be zero if the mapping does not change."
    )
    out: list[Offer] = []
    for factor in GROWTH_FACTORS:
        args = {"cores": list(node.core_ids), "memory": level.name, "factor": factor}
        cost = _price("core.memory.capacity", args, bundle, budget, target, vetoes)
        if cost is None:
            continue
        out.append(
            Offer(
                operator_id="core.memory.capacity",
                tier=OperatorTier.CORE,
                kind="hardware",
                target=target,
                args=args,
                evidence=(
                    f"'{level.name}' stalls {level.stall_cycles:.0f} cycles (veto passes) AND is full: "
                    f"shared capacity utilization {utilization:.3g}"
                ),
                effect=f"multiply the size of '{level.name}' on cores {list(node.core_ids)} by {factor}",
                couples=OPERATORS["core.memory.capacity"].couples,
                predicted_delta=PredictedDelta(cycles=level.stall_cycles, scope="node", derivation=derivation),
                cost=cost,
            )
        )
    return out


def _infeasible_capacity_offers(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    vetoes: list[Veto],
) -> list[Offer]:
    """Capacity growth on an INFEASIBLE run, selected from the typed infeasibility diagnosis.

    Rule 1's veto is about a run that solved and demonstrably did not stall at a level. A solve that
    produced no schedule has no stall vector at all; what it has is an unmet capacity constraint that
    names the resource and the exact shortfall, which is a stronger signal than either.
    """
    if evidence.feasible or bundle is None:
        return []
    out: list[Offer] = []
    for unmet in evidence.unmet_capacity:
        if unmet.resource_kind != "core" or unmet.bound_value <= 0:
            continue
        try:
            core_id = int(unmet.resource_id)
        except ValueError:
            continue
        memory = _memory_matching_bound(bundle, core_id, unmet.bound_value)
        if memory is None:
            vetoes.append(
                Veto(
                    "core.memory.capacity",
                    {"core": core_id, "resource": unmet.resource_label},
                    "rule-1",
                    f"{unmet.resource_label} is over capacity at {unmet.bound_value:.6g} {unmet.unit}, but no "
                    "memory this core declares has that size, so the report cannot be joined to a level; "
                    "growing the wrong one would relieve nothing",
                )
            )
            continue
        # Round the shortfall up to the next declared growth step: the model is not accurate enough
        # to justify a bespoke factor, and a declared step is what the budget was written against.
        needed = unmet.demand_value / unmet.bound_value
        factor = next((f for f in GROWTH_FACTORS if f >= needed), None)
        target = {"core": core_id, "memory": memory, "resource": unmet.resource_label}
        if factor is None:
            vetoes.append(
                Veto(
                    "core.memory.capacity",
                    target,
                    "budget",
                    f"{unmet.resource_label} is short by {needed:.2g}x, beyond the largest declared "
                    f"growth step ({max(GROWTH_FACTORS)}x); shrink the mapping instead",
                )
            )
            continue
        args = {"cores": [core_id], "memory": memory, "factor": factor}
        cost = _price("core.memory.capacity", args, bundle, budget, target, vetoes)
        if cost is None:
            continue
        out.append(
            Offer(
                operator_id="core.memory.capacity",
                tier=OperatorTier.CORE,
                kind="hardware",
                target=target,
                args=args,
                evidence=(
                    f"solve was INFEASIBLE: {unmet.resource_label} demands {unmet.demand_value:.6g} "
                    f"{unmet.unit} against a bound of {unmet.bound_value:.6g} (gap {unmet.gap:.6g}). "
                    f"Levers reported by the solver: {'; '.join(unmet.levers) or 'none'}"
                ),
                effect=f"multiply the size of '{memory}' on core {core_id} by {factor}",
                couples=OPERATORS["core.memory.capacity"].couples,
                predicted_delta=PredictedDelta(
                    cycles=0.0,
                    scope="iteration",
                    derivation=(
                        "an infeasible solve has no latency to improve on; the predicted effect is "
                        "feasibility, not cycles. The shortfall is "
                        f"{unmet.gap:.6g} {unmet.unit} and {factor}x clears it."
                    ),
                ),
                cost=cost,
            )
        )
    return out


# ── Rule 1's exception: coupled array resize ────────────────────────────────────────────────────


def _core_array(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    mapping_params: dict[str, Any],
) -> tuple[list[Offer], list[Veto]]:
    """Resizing a compute array — the one growth Rule 1 does not veto, because it never travels
    alone: the register file that feeds the array is rescaled with it, atomically."""
    del mapping_params
    offers: list[Offer] = []
    vetoes: list[Veto] = []
    if bundle is None:
        return offers, vetoes

    for node in evidence.nodes:
        target = {"node": node.name, "cores": list(node.core_ids)}
        if node.evidence != "cme":
            vetoes.append(
                Veto("core.array.resize", target, "rule-1", f"node '{node.name}' has evidence=none: nothing measured")
            )
            continue
        utilization = node.mac_spatial_utilization
        if utilization is None or utilization < MAC_ARRAY_SATURATED:
            vetoes.append(
                Veto(
                    "core.array.resize",
                    target,
                    "rule-1-exception",
                    f"node '{node.name}' does not fill the array it already has "
                    f"(mac_spatial_utilization={utilization if utilization is not None else 'unknown'}); "
                    "a larger array would only add unused columns",
                )
            )
            continue
        # Rule 2 gates the core tier as a whole. It is conservative here -- a node at exactly
        # compute_efficiency == 1 is the one a bigger array helps most -- but conservative in the
        # safe direction: it withholds an operator, it never licenses one.
        if node.compute_efficiency is not None and node.compute_efficiency >= 1.0:
            vetoes.append(
                Veto(
                    "core.array.resize",
                    target,
                    "rule-2",
                    f"node '{node.name}' shows no discrepancy (compute_efficiency="
                    f"{node.compute_efficiency:.4g}); core-tier operators are reserved for nodes that do",
                )
            )
            continue
        if node.ideal_cycles is None:
            continue
        offers.extend(_array_offers(node, bundle, budget, vetoes))
    return offers, vetoes


def _array_offers(
    node: NodeEvidence,
    bundle: HardwareBundle,
    budget: HardwareBudget | None,
    vetoes: list[Veto],
) -> list[Offer]:
    """One offer per resized dimension, applied to the node's WHOLE core set.

    Per-core edits are expressible (that is what de-aliasing bought) but not useful here: a node is
    inter-core-tiled evenly across these cores, so enlarging one of them leaves the others setting
    the latency and buys nothing.
    """
    core_ids = list(node.core_ids)
    if not core_ids:
        return []
    array = (bundle.cores.get(core_ids[0]) or {}).get("operational_array") or {}
    dims = [str(d) for d in array.get("dimensions") or ()]
    sizes = [int(s) for s in array.get("sizes") or ()]
    if not dims or len(dims) != len(sizes):
        return []

    out: list[Offer] = []
    assert node.ideal_cycles is not None
    for dim, size in zip(dims, sizes, strict=True):
        for factor in ARRAY_SCALE_FACTORS:
            target = {"node": node.name, "cores": core_ids, "dim": dim, "from": size, "to": size * factor}
            args = {"cores": core_ids, "dims": {dim: factor}}
            coupled = _coupled_levels(bundle, core_ids[0], {dim: factor})
            cost = _price("core.array.resize", args, bundle, budget, target, vetoes)
            if cost is None:
                continue
            out.append(
                Offer(
                    operator_id="core.array.resize",
                    tier=OperatorTier.CORE,
                    kind="hardware",
                    target=target,
                    args=args,
                    evidence=(
                        f"node '{node.name}' fills the array (mac_spatial_utilization="
                        f"{node.mac_spatial_utilization:.4g}) at compute_efficiency="
                        f"{node.compute_efficiency:.4g}, so its {node.ideal_cycles:.0f} ideal cycles are "
                        "set by the array size"
                    ),
                    effect=(
                        f"cores {core_ids}: operational_array {dim} {size} -> {size * factor}; "
                        + (
                            "coupled core-local levels " + ", ".join(f"{n} x{s}" for n, s in coupled.items())
                            if coupled
                            else "no core-local level serves this dimension"
                        )
                    ),
                    couples=OPERATORS["core.array.resize"].couples,
                    predicted_delta=PredictedDelta(
                        cycles=node.ideal_cycles * (1.0 - 1.0 / factor),
                        scope="node",
                        derivation=(
                            f"the array does {factor}x more MACs per cycle, so the compute-ideal floor "
                            f"falls from {node.ideal_cycles:.0f} to {node.ideal_cycles / factor:.0f} cycles. "
                            "An upper bound: it is only realised if the node stays compute-bound and the "
                            "spatial mapping still fills the larger array -- which the post-hoc "
                            "mac_spatial_utilization check verifies after the solve."
                        ),
                    ),
                    cost=cost,
                )
            )
    return out


# ── Rule 3: system tier ─────────────────────────────────────────────────────────────────────────


def _system(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    mapping_params: dict[str, Any],
) -> tuple[list[Offer], list[Veto]]:
    """Fusion / intra-core tile / core-count, selected from the binding set and the II decomposition."""
    del budget
    offers: list[Offer] = []
    vetoes: list[Veto] = []
    interval = evidence.initiation_interval
    if interval is None:
        return offers, vetoes

    # RecMII is a hard floor on the II that no reshuffle of resources can move: a loop-carried state
    # forbids the overlap outright.
    if evidence.recurrence_bound_cycles >= interval:
        vetoes.append(
            Veto(
                "system.*",
                {},
                "rule-3",
                f"II = {interval:.0f} cycles is at its recurrence bound "
                f"(RecMII = {evidence.recurrence_bound_cycles:.0f}); no allocation, fusion or tiling "
                "change can overlap iterations that a loop-carried state forbids overlapping",
            )
        )
        return offers, vetoes

    offers.extend(_tiling_offers(evidence, mapping_params, interval, vetoes))
    offers.extend(_fusion_offers(evidence, mapping_params, vetoes))
    offers.extend(_core_count_offers(evidence, bundle, mapping_params, vetoes))
    return offers, vetoes


def _tiling_offers(
    evidence: RunEvidence,
    mapping_params: dict[str, Any],
    interval: float,
    vetoes: list[Veto],
) -> list[Offer]:
    current = _tiling_spec(mapping_params)
    if current is None:
        # The tile is named ``<node>.D<n>`` in the mapper's own namespace, which the solved IR does
        # not report (it reports the global ``z`` dims). With no spec to edit there is no legal
        # target to name, and inventing one would silently be dropped by the per-group filter.
        vetoes.append(
            Veto(
                "system.tiling.intra_core",
                {},
                "rule-3",
                "this run used automatic fusion tiling, so no intra_core_tiling entry names a "
                "dimension to change; the solved IR reports global dims (z*), not the mapper's "
                "<node>.D<n> namespace",
            )
        )
        return []
    transfer = evidence.transfer_bound_cycles or 0.0
    if transfer <= 0:
        vetoes.append(
            Veto(
                "system.tiling.intra_core",
                {},
                "rule-3",
                "no per-iteration cycle is transfer-bound, so a different tile has nothing to hide behind compute",
            )
        )
        return []

    out: list[Offer] = []
    for index, entry in enumerate(current):
        tile = int(entry["tile"])
        for factor in (0.5, 2.0):
            new_tile = int(tile * factor)
            if new_tile < 1 or new_tile == tile:
                continue
            spec = [dict(e) for e in current]
            spec[index]["tile"] = new_tile
            out.append(
                Offer(
                    operator_id="system.tiling.intra_core",
                    tier=OperatorTier.SYSTEM,
                    kind="mapping",
                    target={"dim": entry["dim"], "from": tile, "to": new_tile},
                    args={"intra_core_tiling": spec},
                    evidence=(
                        f"II = {interval:.0f} cycles against RecMII = {evidence.recurrence_bound_cycles:.0f}, "
                        f"and {transfer:.0f} of {evidence.latency_per_iteration:.0f} per-iteration cycles are "
                        "transfer-bound: the schedule is not at its recurrence floor and moves data it "
                        "does not hide"
                    ),
                    effect=f"intra_core_tiling {entry['dim']}: {tile} -> {new_tile}",
                    couples=OPERATORS["system.tiling.intra_core"].couples,
                    predicted_delta=PredictedDelta(
                        cycles=transfer,
                        scope="iteration",
                        derivation=(
                            "upper bound = the transfer-bound cycles of one iteration "
                            f"({transfer:.0f}), i.e. what a tile that overlapped every transfer with "
                            "compute would reclaim. A tile change can also move cycles the other way."
                        ),
                    ),
                )
            )
    return out


def _fusion_offers(evidence: RunEvidence, mapping_params: dict[str, Any], vetoes: list[Veto]) -> list[Offer]:
    """Where to cut the workload. Two directions, each with its own evidence.

    *Cutting* a fused group relieves on-chip residency, and the only measurement that says residency
    is the problem is an infeasible solve with a memory-capacity conflict. *Re-fusing* keeps
    intermediates on chip, and the measurement that says they are not is transfer-bound cycles.
    Neither direction is offered on the strength of "the current arrangement could be different".
    """
    del mapping_params
    if not evidence.fused_group_layers or evidence.n_nodes < MIN_FUSIBLE_LAYERS:
        return []
    transfer = evidence.transfer_bound_cycles or 0.0
    fully_fused = evidence.n_fused_groups == 1
    layers = [layer for group in evidence.fused_group_layers for layer in group]
    target = {"n_groups": evidence.n_fused_groups, "n_layers": evidence.n_nodes}

    if not evidence.feasible and evidence.unmet_capacity and fully_fused:
        unmet = evidence.unmet_capacity[0]
        cut_after = layers[len(layers) // 2 - 1]
        return [
            _fusion_offer(
                target,
                {"fusion_cut_points": [cut_after]},
                evidence_text=(
                    f"the single fused group does not fit: {unmet.resource_label} demands "
                    f"{unmet.demand_value:.6g} {unmet.unit} against {unmet.bound_value:.6g}"
                ),
                effect=f"fusion_cut_points -> ['{cut_after}'] (split the group in two)",
                delta=PredictedDelta(
                    cycles=0.0,
                    scope="iteration",
                    derivation=(
                        "an infeasible solve has no latency to improve on; cutting the group removes "
                        f"the {unmet.gap:.6g} {unmet.unit} of residency it cannot afford. The cost is "
                        "off-chip traffic for the intermediate that stops being fused."
                    ),
                ),
            )
        ]

    if fully_fused:
        vetoes.append(
            Veto(
                "system.fusion.cut",
                target,
                "rule-3",
                "the workload is already one fused group and it solved: nothing measured says the "
                "group is too large to keep on chip, and cutting it can only add off-chip traffic",
            )
        )
        return []
    if transfer <= 0:
        vetoes.append(
            Veto(
                "system.fusion.cut",
                target,
                "rule-3",
                "no per-iteration cycle is transfer-bound, so keeping more intermediates on chip has "
                "no exposed transfer to remove",
            )
        )
        return []
    return [
        _fusion_offer(
            target,
            {"fusion_cut_points": None},
            evidence_text=(
                f"{evidence.n_fused_groups} groups over {evidence.n_nodes} layers, with {transfer:.0f} of "
                f"{evidence.latency_per_iteration:.0f} per-iteration cycles transfer-bound: intermediates "
                "are leaving the chip between groups"
            ),
            effect="fusion_cut_points -> derived from the affine barriers (maximal fusion)",
            delta=PredictedDelta(
                cycles=transfer,
                scope="iteration",
                derivation=(
                    f"upper bound = the transfer-bound cycles of one iteration ({transfer:.0f}); fusing "
                    "can at best keep every intermediate on chip and hide the rest behind compute. "
                    "Re-fusing also raises on-chip pressure and can make a group infeasible."
                ),
            ),
        )
    ]


def _fusion_offer(
    target: dict[str, Any],
    args: dict[str, Any],
    *,
    evidence_text: str,
    effect: str,
    delta: PredictedDelta,
) -> Offer:
    return Offer(
        operator_id="system.fusion.cut",
        tier=OperatorTier.SYSTEM,
        kind="mapping",
        target=target,
        args=args,
        evidence=evidence_text,
        effect=effect,
        couples=OPERATORS["system.fusion.cut"].couples,
        predicted_delta=delta,
    )


def _core_count_offers(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    mapping_params: dict[str, Any],
    vetoes: list[Veto],
) -> list[Offer]:
    """Spread the work wider, when there is idle silicon AND the load is uneven.

    Busy time is derivable: ``busy_i = latency_per_iteration - slack_i``. The classic load-balance
    bound then says the schedule cannot go below the mean busy time however the work is spread, so
    ``max_busy - mean_busy`` is the ceiling on a rebalance.
    """
    per_iteration = evidence.latency_per_iteration
    cores = [s for s in evidence.per_resource_slack if s.kind == "core"]
    if per_iteration is None or not cores or bundle is None:
        return []
    available = _compute_core_ids(bundle)
    current = int(mapping_params.get("nb_cols_to_use") or 0)
    target = {"nb_cols_to_use": current, "compute_cores": len(available)}
    if current >= len(available):
        vetoes.append(
            Veto(
                "system.alloc.cores",
                target,
                "rule-3",
                f"the mapper may already use {current} of {len(available)} compute cores; there is no "
                "wider allocation to ask for",
            )
        )
        return []

    busy = [max(0.0, per_iteration - s.slack_cycles) for s in cores]
    active = [b for b in busy if b > 0]
    if not active:
        return []
    peak, mean = max(active), sum(active) / len(active)
    if peak < mean * LOAD_IMBALANCE:
        vetoes.append(
            Veto(
                "system.alloc.cores",
                target,
                "rule-3",
                f"the busy cores are already balanced (peak {peak:.0f} vs mean {mean:.0f} cycles per "
                "iteration); more cores cannot take work off a resource that is not the outlier",
            )
        )
        return []

    new_value = min(len(available), max(current * 2, current + 1))
    return [
        Offer(
            operator_id="system.alloc.cores",
            tier=OperatorTier.SYSTEM,
            kind="mapping",
            target=target,
            args={"nb_cols_to_use": new_value},
            evidence=(
                f"binding resources {list(evidence.binding_resources)} are busy {peak:.0f} cycles per "
                f"iteration against a {mean:.0f}-cycle mean over the {len(active)} active cores, while "
                f"the mapper is limited to {current} of {len(available)} compute cores"
            ),
            effect=f"nb_cols_to_use: {current} -> {new_value}",
            couples=OPERATORS["system.alloc.cores"].couples,
            predicted_delta=PredictedDelta(
                cycles=peak - mean,
                scope="iteration",
                derivation=(
                    f"busy_i = latency_per_iteration - slack_i; peak {peak:.0f}, mean {mean:.0f}. No "
                    "redistribution can go below the mean, so the ceiling is their difference. It is "
                    "only reachable if the added cores can actually run the binding resource's "
                    "operators -- operator_types may forbid it, in which case the saving is zero."
                ),
            ),
        )
    ]


# ── Rule 4: NoC and off-chip ────────────────────────────────────────────────────────────────────


def _link(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    mapping_params: dict[str, Any],
) -> tuple[list[Offer], list[Veto]]:
    """Bandwidth is the knob that always looks helpful. It is offered only when a link is what the
    solver says caps the overlap, the schedule is not compute-bound, and the saving clears the floor."""
    del mapping_params
    offers: list[Offer] = []
    vetoes: list[Veto] = []
    per_iteration = evidence.latency_per_iteration
    if per_iteration is None or bundle is None:
        return offers, vetoes

    binding_links = evidence.binding_links()
    if not binding_links:
        vetoes.append(
            Veto(
                "link.bandwidth",
                {"binding_resources": list(evidence.binding_resources)},
                "rule-4",
                "no link is in the solver's binding set: every link has strictly more steady-state "
                "slack than the resources that cap the overlap, so its transfers are already "
                "overlapped and widening it removes no exposed cycle",
            )
        )
        return offers, vetoes

    compute_pct = 100.0 * (evidence.compute_bound_cycles or 0.0) / per_iteration
    if compute_pct >= COMPUTE_BOUND_PCT:
        vetoes.append(
            Veto(
                "link.bandwidth",
                {"links": [s.resource for s in binding_links]},
                "rule-4",
                f"the schedule is compute-bound ({compute_pct:.1f}% of per-iteration cycles); a wider "
                "link cannot shorten a slot whose latency is set by the array",
            )
        )
        return offers, vetoes

    floor = evidence.noise_floor
    for slack in binding_links:
        exposed = max(0.0, per_iteration - slack.slack_cycles)
        for factor in GROWTH_FACTORS:
            saving = min(evidence.transfer_bound_cycles or exposed, exposed * (1.0 - 1.0 / factor))
            target = {"link": slack.resource, "exposed_cycles": exposed}
            if not floor.clears(saving):
                vetoes.append(
                    Veto(
                        "link.bandwidth",
                        {**target, "factor": factor},
                        "rule-4",
                        f"predicted saving {saving:.0f} cycles does not clear the noise floor "
                        f"({floor.cycles:.0f} cycles; {floor.source})",
                    )
                )
                continue
            args = {"link_bandwidth": _link_bandwidth(slack.resource), "factor": factor}
            if args["link_bandwidth"] is None:
                vetoes.append(
                    Veto(
                        "link.bandwidth",
                        target,
                        "rule-4",
                        f"the solver names this resource '{slack.resource}', which carries no bandwidth "
                        "to join against the bundle's core_connectivity; the link cannot be identified",
                    )
                )
                continue
            cost = _price("link.bandwidth", args, bundle, budget, target, vetoes)
            if cost is None:
                continue
            offers.append(
                Offer(
                    operator_id="link.bandwidth",
                    tier=OperatorTier.LINK,
                    kind="hardware",
                    target=target,
                    args=args,
                    evidence=(
                        f"'{slack.resource}' is in the solver's binding set with {slack.slack_cycles:.0f} "
                        f"cycles of slack, i.e. {exposed:.0f} exposed cycles per iteration, and only "
                        f"{compute_pct:.1f}% of the iteration is compute-bound"
                    ),
                    effect=f"multiply the bandwidth of every {args['link_bandwidth']} bits/cycle link by {factor}",
                    couples=OPERATORS["link.bandwidth"].couples,
                    predicted_delta=PredictedDelta(
                        cycles=saving,
                        scope="iteration",
                        derivation=(
                            f"{factor}x bandwidth cuts the link's {exposed:.0f} exposed cycles to "
                            f"{exposed / factor:.0f}, capped by the "
                            f"{evidence.transfer_bound_cycles or exposed:.0f} transfer-bound cycles the "
                            f"iteration actually has. Clears the noise floor of {floor.cycles:.0f} cycles "
                            f"({floor.source})."
                        ),
                    ),
                    cost=cost,
                )
            )
    return offers, vetoes


# ── Applying ────────────────────────────────────────────────────────────────────────────────────


def apply_operator(
    operator_id: str,
    args: dict[str, Any],
    *,
    bundle: HardwareBundle | None = None,
    mapping_params: dict[str, Any] | None = None,
) -> AppliedOperator:
    """Apply one operator. The single implementation of every ``effect`` in the registry."""
    operator = OPERATORS.get(operator_id)
    if operator is None:
        raise KeyError(f"Unknown operator '{operator_id}'. Known: {sorted(OPERATORS)}")
    params = dict(mapping_params or {})
    notes: list[str] = []

    if operator.kind == "mapping":
        for key in ("intra_core_tiling", "fusion_cut_points", "nb_cols_to_use", "pipelining"):
            if key in args:
                params[key] = args[key]
                notes.append(f"{key} = {args[key]!r}")
        return AppliedOperator(operator_id, bundle, params, notes)

    if bundle is None:
        raise ValueError(f"Operator '{operator_id}' edits hardware but no bundle was given")
    edited = bundle.copy()
    if operator_id == "core.memory.bandwidth":
        notes += _scale_memory(edited, args["cores"], args["memory"], bandwidth=int(args["factor"]))
    elif operator_id == "core.memory.capacity":
        notes += _scale_memory(edited, args["cores"], args["memory"], capacity=int(args["factor"]))
    elif operator_id == "core.array.resize":
        notes += _resize_array(edited, args["cores"], {str(d): int(f) for d, f in args["dims"].items()})
    elif operator_id == "link.bandwidth":
        notes += _scale_links(edited, int(args["link_bandwidth"]), int(args["factor"]))
    return AppliedOperator(operator_id, edited, params, notes)


def _scale_memory(
    bundle: HardwareBundle,
    core_ids: list[int],
    memory: str,
    *,
    capacity: int = 1,
    bandwidth: int = 1,
    follow_aliases: bool = True,
) -> list[str]:
    """Scale one memory level's capacity and/or port widths on the given cores.

    Aliased views are carried along by default. The MXU's ``operand_buffer``, the VPU's and the VMEM
    core's ``vmem`` are three views of one 128 MiB scratchpad; widening only the view the stalling
    node happens to sit behind would give the solve a wider port that the cost model never bills and
    the other views never see -- a mutation that is free precisely because it is incoherent.
    """
    targets = {(int(c), memory) for c in core_ids}
    if follow_aliases:
        for group in _alias_groups(bundle):
            if targets & group:
                targets |= group
    notes: list[str] = []
    for core_id, mem_name in sorted(targets):
        core = bundle.cores.get(core_id)
        mem = ((core or {}).get("memories") or {}).get(mem_name)
        if mem is None:
            continue
        if capacity > 1:
            mem["size"] = int(mem.get("size", 0)) * capacity
            notes.append(f"core {core_id}: {mem_name}.size x{capacity} -> {mem['size']}")
        if bandwidth > 1:
            for port in mem.get("ports") or ():
                for key in ("bandwidth_min", "bandwidth_max"):
                    if key in port:
                        port[key] = int(port[key]) * bandwidth
            notes.append(f"core {core_id}: {mem_name} port bandwidths x{bandwidth}")
    return notes


def _alias_groups(bundle: HardwareBundle) -> list[set[tuple[int, str]]]:
    """The bundle's ``memory_aliases`` as ``(core id, memory name)`` sets."""
    groups: list[set[tuple[int, str]]] = []
    for group in bundle.memory_aliases:
        refs: set[tuple[int, str]] = set()
        for ref in group:
            core_id, _, name = str(ref).partition(".")
            try:
                refs.add((int(core_id), name))
            except ValueError:
                continue
        if refs:
            groups.append(refs)
    return groups


def _shared_memories(bundle: HardwareBundle) -> set[tuple[int, str]]:
    """``(core id, memory name)`` pairs the bundle declares as views of ONE physical memory."""
    return {ref for group in _alias_groups(bundle) for ref in group}


def _coupled_levels(bundle: HardwareBundle, core_id: int, dim_factors: dict[str, int]) -> dict[str, int]:
    """The per-level scale factors an array resize must carry with it.

    Per level, not globally, and only for CORE-LOCAL levels. Applied blindly, doubling a 256x256
    array along both axes would ask a *shared* 128 MiB scratchpad for 4x its port width -- on the
    TPU7x that is 1,048,576 bits/cycle, about 228 TB/s and four times the modelled part. The array
    belongs to one core; the scratchpad does not, so it is never scaled here. It is identified from
    the bundle's own ``memory_aliases``, not from a name heuristic.

    A level's factor is the product of the resize factors over the dimensions it actually serves.
    A level with ``served_dimensions: []`` (a per-PE register file) is served along no dimension,
    which means it is *replicated* across every one of them: its instance count already follows the
    array in the cost model, and scaling its capacity too would double-count the same silicon.
    """
    shared = _shared_memories(bundle)
    core = bundle.cores.get(int(core_id)) or {}
    out: dict[str, int] = {}
    for name, mem in (core.get("memories") or {}).items():
        if (int(core_id), name) in shared:
            continue
        served = {str(d) for d in mem.get("served_dimensions") or ()}
        scale = math.prod(factor for dim, factor in dim_factors.items() if dim in served)
        if scale > 1:
            out[name] = scale
    return out


def _resize_array(bundle: HardwareBundle, core_ids: list[int], dim_factors: dict[str, int]) -> list[str]:
    notes: list[str] = []
    for core_id in core_ids:
        core = bundle.cores.get(int(core_id))
        array = (core or {}).get("operational_array") or {}
        dims = [str(d) for d in array.get("dimensions") or ()]
        sizes = list(array.get("sizes") or ())
        for dim, factor in dim_factors.items():
            if dim not in dims:
                continue
            index = dims.index(dim)
            sizes[index] = int(sizes[index]) * factor
            notes.append(f"core {core_id}: array {dim} -> {sizes[index]}")
        array["sizes"] = sizes
        for name, scale in _coupled_levels(bundle, core_id, dim_factors).items():
            # follow_aliases=False is the point of the whole exception: `_coupled_levels` has already
            # excluded shared levels, and re-admitting them through the alias walk would be the
            # 228 TB/s scratchpad demand this rule exists to prevent.
            notes += _scale_memory(bundle, [core_id], name, capacity=scale, bandwidth=scale, follow_aliases=False)
    return notes


def _scale_links(bundle: HardwareBundle, bandwidth: int, factor: int) -> list[str]:
    notes: list[str] = []
    for entry in bundle.accelerator.get("core_connectivity") or ():
        if int(entry.get("bandwidth", 0)) == bandwidth:
            entry["bandwidth"] = bandwidth * factor
            notes.append(f"{entry.get('type')} {entry.get('cores')}: bandwidth {bandwidth} -> {bandwidth * factor}")
    return notes


# ── The post-hoc guard for D3 ───────────────────────────────────────────────────────────────────


def post_hoc_check(operator_id: str, target: dict[str, Any], before: RunEvidence, after: RunEvidence) -> str | None:
    """Verify after the solve that a coupled resize actually held. None = accepted.

    ZigZag's validator checks none of this and degrades silently — ``limit_unrolling_to_mem_bandwidth``
    merely logs when the unrolling it was asked for does not fit the memory bandwidth. So the only
    reliable test that the coupling was right is the outcome: if the node's spatial utilization fell,
    the larger array is not being fed and the mutation must be rejected rather than scored on latency
    it did not earn.
    """
    if operator_id != "core.array.resize":
        return None
    name = target.get("node")
    old, new = before.node(str(name)), after.node(str(name))
    if old is None or new is None:
        return None
    if old.mac_spatial_utilization is None or new.mac_spatial_utilization is None:
        return (
            f"node '{name}' reports no mac_spatial_utilization after the resize, so the coupling "
            "cannot be verified; rejecting rather than crediting an unverified change"
        )
    if new.mac_spatial_utilization < old.mac_spatial_utilization:
        return (
            f"node '{name}' mac_spatial_utilization fell {old.mac_spatial_utilization:.4g} -> "
            f"{new.mac_spatial_utilization:.4g}: the resized array is not being fed, so the coupled "
            "memory rescale was wrong for this core"
        )
    return None


# ── Helpers ─────────────────────────────────────────────────────────────────────────────────────


def _price(  # noqa: PLR0913 -- the guard needs all of them; splitting it would only move arguments
    operator_id: str,
    args: dict[str, Any],
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    target: dict[str, Any],
    vetoes: list[Veto],
) -> dict[str, Any] | None:
    """Price the bundle this edit would produce; None (with a veto) when it busts the budget.

    This runs before anything is launched, which is the point: the engine's own ``unit_area: 0``
    means a solve would happily report the mutated hardware as free.
    """
    if bundle is None:
        return None
    edited = apply_operator(operator_id, args, bundle=bundle).bundle
    assert edited is not None
    try:
        verdict = check_budget(edited, budget) if budget else None
    except ValueError as exc:
        # A mutated bundle that no longer validates as an accelerator must not reach a solve.
        vetoes.append(Veto(operator_id, target, "budget", f"the edited bundle is not a valid accelerator: {exc}"))
        return None
    if verdict is None:
        return None
    if not verdict.ok:
        vetoes.append(Veto(operator_id, target, "budget", f"rejected before any solve: {verdict}"))
        return None
    return {
        "area_mm2": verdict.report.total_area_mm2,
        "peak_access_energy_pj_per_cycle": verdict.report.peak_access_energy_pj_per_cycle,
        "budget": budget.label if budget else None,
        "max_area_mm2": budget.max_area_mm2 if budget else None,
    }


BITS_PER_BYTE = 8


def _memory_matching_bound(bundle: HardwareBundle, core_id: int, bound_value: float) -> str | None:
    """The level an infeasibility report is talking about, joined by its declared capacity.

    The report names the resource (a core) but not the level; its ``bound_value`` IS the level's
    size, so the join is exact rather than a guess at "the biggest one". Bytes and bits are both
    accepted because the unit is the report's choice. None when nothing matches — growing a level
    the conflict is not about would relieve nothing.
    """
    memories = (bundle.cores.get(core_id) or {}).get("memories") or {}
    for name, mem in memories.items():
        size = int(mem.get("size", 0))
        if size and (size == int(bound_value) or size == int(bound_value) * BITS_PER_BYTE):
            return name
    return None


def _compute_core_ids(bundle: HardwareBundle) -> list[int]:
    """Cores with a non-empty operational array — the ones a wider allocation could reach."""
    out = []
    for core_id, core in bundle.cores.items():
        sizes = (core.get("operational_array") or {}).get("sizes") or ()
        if math.prod(int(s) for s in sizes) > 1:
            out.append(int(core_id))
    return sorted(out)


def _tiling_spec(mapping_params: dict[str, Any]) -> list[dict[str, Any]] | None:
    spec = mapping_params.get("intra_core_tiling")
    if not isinstance(spec, list) or not spec:
        return None
    entries = [e for e in spec if isinstance(e, dict) and "dim" in e and "tile" in e]
    return copy.deepcopy(entries) or None


def _link_bandwidth(resource: str) -> int | None:
    """The solver spells a link ``CL(Any, Any, bw=65536)``; the bandwidth is the only part of that
    key that joins to the bundle's ``core_connectivity``. None when it carries none."""
    marker = "bw="
    if marker not in resource:
        return None
    digits = "".join(c for c in resource.split(marker, 1)[1] if c.isdigit())
    return int(digits) if digits else None
