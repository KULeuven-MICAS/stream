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
    An expected improvement *with its derivation and its units*, always an upper bound — cycles for
    a growth, mm² for a reduction. This is what makes a proposal falsifiable: the run that follows
    either recovers it or does not, and :mod:`stream.dse.residual` scores which.

THE FOUR RULES, AND THE DUAL
----------------------------
1. **The veto.** No stall at a memory level ⇒ growing that level's size *or* its bandwidth is
   illegal: absent a stall, neither changes latency there. The stall vector is a bandwidth quantity
   by construction (``real_cycle`` is bandwidth-derived), so it can *select* a bandwidth growth but
   not a capacity one — capacity is selected from the infeasibility report or a saturated
   ``mem_utili_shared``. And ``evidence: "none"`` is never "no stall": a missing CME is a missing
   measurement, and the operator is simply not offered.

   **Rule 1's dual** reads the same measurements the other way: capacity above the solved working
   set, and port width above the measured occupancy, are silicon this workload never uses, and
   giving it back is a design result rather than a consolation prize. The guards are the mirror
   image — an unmeasured memory is never cut, no declared step may cross the measured working set,
   and a shrink that makes the mapping infeasible is a regression, not a win.
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
everything", because the engine's own ``unit_area: 0`` makes growth free. The
:class:`~stream.dse.objective.Objective` then filters what is left: a move that cannot improve the
declared objective is vetoed with that as the stated reason, so a latency search is never offered an
area saving and an area search is never offered a way to spend silicon.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from stream.dse.evidence import (
    MemoryLevelEvidence,
    NodeEvidence,
    NoiseFloor,
    RunEvidence,
)
from stream.dse.objective import Objective, ObjectiveKind
from stream.dse.residual import OperatorScorecard
from stream.hardware.bundle import HardwareBundle
from stream.hardware.cost import HardwareBudget, check_budget, evaluate_bundle_cost

# ── Tunables ────────────────────────────────────────────────────────────────────────────────────

DEFAULT_BUDGET_HEADROOM = 0.10
"""Fraction of the baseline area/power a variant may exceed when no explicit budget is given."""

GROWTH_FACTORS: tuple[int, ...] = (2, 4)
"""Declared range for a single growth step. A step budget, not a free variable: an operator that
could ask for any factor would simply ask for the largest one that still fits the budget."""

SHRINK_BANKS = 8
"""Granularity a capacity may be resized at, as a fraction of the memory's declared size.

A large SRAM is built from banks, so its capacity is quantised: a design drops a bank, not an
arbitrary number of bits. An eighth is a conservative stand-in for one — 16 MiB of a 128 MiB
scratchpad, 8 KB of a 64 KB tile memory.

The growth operators step by *doublings*, and the same step in reverse would be useless here.
A fused workload routinely leaves a scratchpad 70-80% full, which is a fifth of a die doing
nothing and a halving away from legal; a rule that could only offer halvings would report that
memory as not over-provisioned and give the silicon back to nobody.
"""

SHRINK_STEPS_OFFERED = 2
"""How many capacities to offer per memory: the tightest legal one and the next size up.

Not every legal bank count. Seven near-identical offers on one memory would make one decision look
like seven pieces of evidence, and the interesting choice is only ever "as small as the measurement
allows" versus "one bank of margin over that".
"""

SHRINK_HEADROOM = 1.10
"""Margin a reduced capacity keeps over the measured working set.

The measurement is exact for the placement that produced it -- and that placement is re-derived on
the smaller memory. Sizing exactly to the old working set therefore spends a whole solve to discover
that the new one is a few tiles bigger. 10% is under one bank for any memory around half full, so
the margin usually costs nothing at all and never costs more than one step of the saving.
"""

NARROW_FACTORS: tuple[int, ...] = (2, 4)
"""Declared divisors for a port-width reduction. Powers of two, unlike the capacity steps: a port is
a physical bus width, so halving it is a real design and shaving an eighth off it is not."""

PORT_OCCUPANCY_CEILING = 0.9
"""Occupancy a port may reach after being narrowed.

Below 1.0 by design: occupancy is measured over ONE node's computation span, and a port driven to
exactly its measured occupancy has no room for the transfer that another node's span overlaps into
it. 0.9 is the margin; a port that cannot keep it does not get the offer.
"""

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

COMPUTE_DOMINATED = 0.9
"""``compute_efficiency`` above which a node's latency is set by its array rather than by stalls.

A full array is necessary for a resize to help but not sufficient: a node can fill its array
spatially and still spend most of its cycles waiting on memory (SwiGLU's ``Elt_Mul`` fills the VPU
and runs at 0.39 efficiency). Enlarging the array there shortens a term that is not the one setting
the latency; the memory operators are what address it.
"""

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
    """An expected improvement, always an upper bound, always with its units spelled out.

    Two units, because the registry now offers moves in both directions. A grow buys **cycles**; a
    shrink buys **mm²** and buys exactly zero cycles. Forcing the second into the first would either
    fabricate a cycle saving it does not have or report it as "no predicted effect", and neither is
    what a design that gives back 70% of its die is doing.
    """

    value: float
    scope: Literal["node", "iteration", "bundle"]
    """"node": cycles off ONE node's contribution to ONE steady-state iteration. ZigZag's
    ``stall_or_slack`` already carries ``(period_count - 1)``, so this is a whole-node figure and a
    consumer must NOT multiply by an iteration count again. "iteration": cycles off one steady-state
    iteration of the whole schedule. "bundle": a whole-accelerator figure, e.g. total die area."""
    derivation: str
    unit: Literal["cycles", "mm2"] = "cycles"
    bound: Literal["upper"] = "upper"

    @property
    def cycles(self) -> float:
        """The predicted cycle saving. 0.0 for a delta measured in another unit.

        0.0 and not ``None``: an area reduction really does buy zero cycles, so every consumer that
        compares this against the cycle noise floor gets the true answer rather than a placeholder
        it has to special-case.
        """
        return self.value if self.unit == "cycles" else 0.0

    @property
    def area_mm2(self) -> float:
        """The predicted area saving in mm². 0.0 for a delta measured in another unit."""
        return self.value if self.unit == "mm2" else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            # `cycles` is kept alongside `value` so a consumer written against the cycles-only
            # registry keeps reading a correct number rather than a renamed one.
            "cycles": self.cycles,
            "value": self.value,
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
            id="core.memory.shrink",
            tier=OperatorTier.CORE,
            kind="hardware",
            summary="Cut the capacity of a memory the solved placement never fills, giving the area back.",
            couples=(
                ALIAS_COUPLING,
                "the floor is the LARGEST residency over the whole alias group and over every fused "
                "group: the memory has to hold whichever of them needs the most",
            ),
        ),
        Operator(
            id="core.memory.narrow",
            tier=OperatorTier.CORE,
            kind="hardware",
            summary="Narrow the ports of a memory level with measured slack and spare port occupancy.",
            couples=(ALIAS_COUPLING, "never below the level's declared bandwidth_min"),
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


# ── Where a judgement's inputs came from ────────────────────────────────────────────────────────
#
# `evidence` is prose. It is what a human reads, and it is unwalkable: "'vregs' stalls 1326 cycles"
# names neither the artifact that measured it nor the path inside it. `refs` is the same claim in a
# shape a consumer can follow, and it has exactly two shapes -- a *fact* in another artifact, or a
# *declared* value that terminates the chain because nobody computed it.

ALLOCATION_ARTIFACT = "allocation.json"
PROGRESS_ARTIFACT = "progress.json"
INFEASIBILITY_ARTIFACT = "infeasibility.json"

# The dominant fused group's own paths. `*` rather than an index because the group whose latency
# sets the runtime is chosen at read time; the static provenance map is keyed by the same glob.
_GROUP = "/groups/*/allocation"
NODE_PERFORMANCE_PATH = f"{_GROUP}/performance/nodes"
OVERLAP_PATH = f"{_GROUP}/performance/overlap"
LATENCY_PATH = f"{_GROUP}/latency"
CORE_COST_NODES_PATH = "/stages/core_cost/artifact/groups/*/nodes"


def fact_ref(artifact: str, path: str) -> dict[str, str]:
    """A reference to another stamped fact: which artifact, and where inside it."""
    return {"kind": "fact", "artifact": artifact, "path": path}


def declared_ref(label: str, path: str | None = None) -> dict[str, str]:
    """The terminus of a chain: a bundle YAML value or a launch parameter nobody computed.

    `declared` is not a producer. It is where every chain ends -- the point past which there is no
    upstream computation to attribute, only somebody's decision.
    """
    ref = {"kind": "declared", "label": label}
    if path is not None:
        ref["path"] = path
    return ref


def node_ref(node_name: str) -> dict[str, str]:
    return fact_ref(PROGRESS_ARTIFACT, f"{CORE_COST_NODES_PATH}/{node_name}")


def node_performance_ref(node_name: str) -> dict[str, str]:
    return fact_ref(ALLOCATION_ARTIFACT, f"{NODE_PERFORMANCE_PATH}/{node_name}")


def slack_ref() -> dict[str, str]:
    return fact_ref(ALLOCATION_ARTIFACT, f"{OVERLAP_PATH}/per_resource_slack")


def binding_ref() -> dict[str, str]:
    return fact_ref(ALLOCATION_ARTIFACT, f"{OVERLAP_PATH}/binding_resources")


def per_iteration_ref() -> dict[str, str]:
    return fact_ref(ALLOCATION_ARTIFACT, f"{LATENCY_PATH}/per_iteration")


def bundle_ref(what: str) -> dict[str, str]:
    return declared_ref(f"hardware bundle: {what}")


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
    refs: tuple[dict[str, str], ...] = ()
    """:attr:`evidence`, walkable. See the module section above. Every chain ends in a `declared`."""

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
            "refs": [dict(ref) for ref in self.refs],
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
    unknown: bool = False
    """True when the refusal is for *absent* evidence rather than for evidence that ruled the move
    out -- ``evidence: "none"`` on a node ZigZag never modelled.

    The two are opposite statements about the design space. "We know this cannot help" is a finding;
    "we could not tell" is a gap in the measurement, and a refusal ledger that reports them as one
    number overstates how much of the space has actually been ruled out.
    """
    refs: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator_id,
            "target": self.target,
            "rule": self.rule,
            "reason": self.reason,
            "unknown": self.unknown,
            "refs": [dict(ref) for ref in self.refs],
        }


@dataclass(frozen=True)
class OfferResult:
    offered: list[Offer]
    vetoed: list[Veto]
    noise_floor: NoiseFloor
    budget_label: str | None = None
    objective: Objective = field(default_factory=Objective)
    scorecard: OperatorScorecard = field(default_factory=OperatorScorecard)

    def as_dict(self) -> dict[str, Any]:
        return {
            # `clears_noise_floor` is reported on every offer, not just the link tier Rule 4 gates on
            # it. A core-level move that removes a real stall worth 463 cycles on a 187,082-cycle
            # iteration is legal and is still invisible end to end; a ranker has to be able to see
            # that, and the guards deliberately do not decide it.
            "offered": [
                {
                    **offer.as_dict(),
                    "clears_noise_floor": self._clears_floor(offer),
                    # E1: every offer says whether it can move THIS objective at all, and E2 says
                    # how much of its prediction its own history has earned.
                    "trust": self.scorecard.trust(offer.operator_id),
                    "discounted_delta": self.scorecard.discounted(offer.operator_id, offer.predicted_delta.value),
                }
                for offer in self.offered
            ],
            "vetoed": [v.as_dict() for v in self.vetoed],
            "noise_floor": {
                "cycles": self.noise_floor.cycles,
                "relative": self.noise_floor.relative,
                "known": self.noise_floor.known,
                "source": self.noise_floor.source,
            },
            "budget": self.budget_label,
            "objective": self.objective.as_dict(),
            "scorecard": self.scorecard.as_dict(),
        }

    def _clears_floor(self, offer: Offer) -> bool:
        """Whether this offer's predicted improvement is distinguishable from noise.

        The noise floor is a *cycle* quantity — it comes from the solver's optimality gap. An area
        saving is not measured by the solver at all: it is arithmetic over the bundle, exact to the
        cost model, and a 121 mm² reduction is not "within the solver's tolerance". So an offer
        denominated in mm² clears by construction, and saying otherwise would report every shrink as
        invisible.
        """
        if offer.predicted_delta.unit != "cycles":
            return offer.predicted_delta.value > 0
        return self.noise_floor.clears(offer.predicted_delta.cycles)

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
    objective: Objective | None = None,
    scorecard: OperatorScorecard | None = None,
) -> OfferResult:
    """Every legal move on this evidence, in Rule 2's tier order, plus what was refused and why.

    ``bundle`` is the hardware the evidence was measured on. Without it no hardware operator can be
    offered at all: its edit could not be priced, and an unpriced edit cannot be certified within
    budget. That is the same discipline as ``evidence: "none"`` — we do not offer what we cannot
    check.

    ``objective`` says what "better" means for this run. It is a *filter*, not a ranking: a move
    that cannot move the declared objective in the right direction is vetoed with that as the
    reason, so a latency search is never handed an area saving and an area search is never handed a
    move that spends silicon. ``scorecard`` carries what earlier applications of each operator
    actually delivered; it only annotates, so a badly-calibrated operator is deprioritised rather
    than removed from the action space.
    """
    mapping_params = dict(mapping_params or {})
    if budget is None and bundle is not None:
        budget = HardwareBudget.from_bundle(bundle, DEFAULT_BUDGET_HEADROOM)
    if objective is None:
        objective = Objective.from_baseline(
            ObjectiveKind.LATENCY,
            baseline_latency_cycles=evidence.latency_total,
            baseline_area_mm2=_bundle_area(bundle),
        )
    offers: list[Offer] = []
    vetoes: list[Veto] = []

    for propose in (_core_memory, _core_shrink, _core_array, _system, _link):
        found, refused = propose(evidence, bundle, budget, mapping_params)
        offers.extend(found)
        vetoes.extend(refused)

    offers = _dedupe(offers)
    offers, refused = _filter_by_objective(offers, objective, feasible=evidence.feasible)
    vetoes.extend(refused)
    # Rule 2 is an ordering over tiers, not over predicted sizes: a core-level fix that removes a
    # measured stall is considered before a system-level reshuffle even when the latter predicts more.
    offers.sort(key=lambda o: OFFER_TIERS.index(o.tier))
    return OfferResult(
        offered=offers,
        vetoed=vetoes,
        noise_floor=evidence.noise_floor,
        budget_label=budget.label if budget else None,
        objective=objective,
        scorecard=scorecard or OperatorScorecard(),
    )


def _filter_by_objective(
    offers: list[Offer], objective: Objective, *, feasible: bool
) -> tuple[list[Offer], list[Veto]]:
    """Drop the moves that cannot improve the declared objective, saying so.

    A latency search offered a 121 mm² area saving, or an area search offered a move that doubles a
    memory, is being invited to spend a wave on something that cannot change its score. Reporting it
    as a veto rather than silently ranking it last is what stops the next wave proposing it again.

    The one exception is an INFEASIBLE run. There is no objective value to improve on a design that
    produced no schedule, so every legal repair stays on the menu whatever the objective is — a
    search that refused to spend area on feasibility would simply never solve.
    """
    if not feasible:
        return offers, []
    kept: list[Offer] = []
    vetoes: list[Veto] = []
    for offer in offers:
        saves_cycles = offer.predicted_delta.cycles > 0
        saves_area = offer.predicted_delta.area_mm2 > 0
        costs_area = offer.kind == "hardware" and not saves_area
        admissible = {
            ObjectiveKind.LATENCY: saves_cycles,
            ObjectiveKind.AREA: saves_area,
            ObjectiveKind.EFFICIENCY: saves_cycles or saves_area,
        }[objective.kind]
        if admissible:
            kept.append(offer)
            continue
        vetoes.append(
            Veto(
                offer.operator_id,
                offer.target,
                "objective",
                f"the declared objective is '{objective.kind}' ({objective.improves_with}), and this move "
                + (
                    f"buys {offer.predicted_delta.value:.4g} {offer.predicted_delta.unit}"
                    if offer.predicted_delta.value > 0
                    else "buys nothing measurable"
                )
                + (", spending silicon rather than saving it" if costs_area else ""),
            )
        )
    return kept, vetoes


def _bundle_area(bundle: HardwareBundle | None) -> float | None:
    """The bundle's modelled die area, or None when it cannot be priced."""
    if bundle is None:
        return None
    try:
        return evaluate_bundle_cost(bundle).total_area_mm2
    except ValueError:
        return None


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
                    unknown=True,
                    refs=(node_ref(node.name),),
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
            predicted_delta=PredictedDelta(value=headroom, scope="node", derivation=derivation),
            cost=cost,
            refs=(
                node_ref(node.name),
                node_performance_ref(node.name),
                bundle_ref(f"port geometry of '{level.name}' on cores {list(node.core_ids)}"),
            ),
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
                predicted_delta=PredictedDelta(value=level.stall_cycles, scope="node", derivation=derivation),
                cost=cost,
                refs=(
                    node_ref(node.name),
                    bundle_ref(f"declared size of '{level.name}' on cores {list(node.core_ids)}"),
                ),
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
                    value=0.0,
                    scope="iteration",
                    derivation=(
                        "an infeasible solve has no latency to improve on; the predicted effect is "
                        "feasibility, not cycles. The shortfall is "
                        f"{unmet.gap:.6g} {unmet.unit} and {factor}x clears it."
                    ),
                ),
                cost=cost,
                refs=(
                    fact_ref(INFEASIBILITY_ARTIFACT, "/resources/*/unmet"),
                    bundle_ref(f"declared size of '{memory}' on core {core_id}"),
                ),
            )
        )
    return out


# ── Rule 1's dual: shrink what the measurement says is over-provisioned ─────────────────────────


@dataclass(frozen=True)
class _MemoryClass:
    """One physical memory design, as it appears on every core that owns an instance of it."""

    name: str
    size_bits: int
    owner_cores: tuple[int, ...]
    """Cores that own an instance — the ones a shrink names."""
    members: tuple[tuple[int, str], ...]
    """Every ``(core, memory)`` view of those instances, alias closure included."""

    @property
    def member_cores(self) -> set[int]:
        return {core_id for core_id, _ in self.members}


@dataclass(frozen=True)
class _WorkingSet:
    """The measured floor under a capacity, and where the measurement came from."""

    bits: int
    sources: tuple[str, ...]

    @property
    def measured(self) -> bool:
        return bool(self.sources)


def _core_shrink(
    evidence: RunEvidence,
    bundle: HardwareBundle | None,
    budget: HardwareBudget | None,
    mapping_params: dict[str, Any],
) -> tuple[list[Offer], list[Veto]]:
    """The dual of Rule 1: a memory the workload demonstrably never fills is over-provisioned.

    Rule 1 says that absent a stall, growing a level cannot make it faster. Read the other way round,
    the same measurements say that the capacity above the solved working set and the port width above
    the measured occupancy are silicon this workload never uses — and *that* is a design change worth
    making on a part already at 98.9% of its own compute roofline, where there is ~1% of latency to
    win and most of the die is scratchpad.

    The guards are the mirror image of the growth guards, not a relaxation of them:

    * **Unknown is not "safe to cut".** A memory with no occupancy measurement is never offered. Note
      what counts as a measurement: the ALLOCATOR's solved residency is one, and it exists even for a
      core whose CME is missing, because it comes from the MILP rather than from ZigZag. So
      ``evidence: "none"`` blocks the ZigZag half of the signal, not the operator outright.
    * **Never below the measured working set.** The floor is the largest residency over the whole
      alias closure and over every fused group, and each declared step is checked against it
      individually. Crossing it does not make the design smaller, it makes it infeasible.
    * **A shrink that breaks the mapping is a regression, not a win.** The floor is arithmetic and
      exact for the placement that was measured, but the temporal mapping is re-derived on the
      smaller memory and may come back worse. That is what the objective's latency ceiling and the
      post-hoc check are for; neither is optional.
    """
    del mapping_params
    offers: list[Offer] = []
    vetoes: list[Veto] = []
    if bundle is None:
        return offers, vetoes

    for memory_class in _memory_classes(bundle):
        target = {
            "memory": memory_class.name,
            "cores": list(memory_class.owner_cores),
            "size_bits": memory_class.size_bits,
        }
        working_set = _measured_working_set(evidence, bundle, memory_class)
        if not working_set.measured:
            vetoes.append(
                Veto(
                    "core.memory.shrink",
                    target,
                    "rule-1-dual",
                    f"nothing measured how full '{memory_class.name}' gets: neither the allocator's solved "
                    "residency nor a CME capacity utilization covers it. An unmeasured memory is not a safe "
                    "one to cut — it is one nothing is known about",
                )
            )
            continue
        if working_set.bits <= 0:
            vetoes.append(
                Veto(
                    "core.memory.shrink",
                    target,
                    "rule-1-dual",
                    f"'{memory_class.name}' holds nothing at all in the solved steady state "
                    f"({'; '.join(working_set.sources)}). A memory this workload never touches says nothing "
                    "about how small it could be for the workloads that do",
                )
            )
            continue
        offers.extend(_shrink_offers(memory_class, working_set, target, bundle, budget, vetoes))
        offers.extend(_narrow_offers(evidence, memory_class, target, bundle, budget, vetoes))
    return offers, vetoes


def _shrink_offers(  # noqa: PLR0913 -- one call site; each argument is an input to the guard
    memory_class: _MemoryClass,
    working_set: _WorkingSet,
    target: dict[str, Any],
    bundle: HardwareBundle,
    budget: HardwareBudget | None,
    vetoes: list[Veto],
) -> list[Offer]:
    """Capacity reduction: size the memory to the measured working set, at bank granularity.

    The floor is arithmetic and exact — the smallest whole number of banks that holds the measured
    residency with :data:`SHRINK_HEADROOM` of margin. Anything below it does not hold the mapping,
    which is why nothing below it is ever offered.
    """
    occupancy = working_set.bits / memory_class.size_bits
    bank_bits = memory_class.size_bits // SHRINK_BANKS
    if bank_bits <= 0:
        return []  # a memory smaller than its own granularity has no smaller size to name
    needed = working_set.bits * SHRINK_HEADROOM
    smallest = math.ceil(needed / bank_bits)
    if smallest >= SHRINK_BANKS:
        vetoes.append(
            Veto(
                "core.memory.shrink",
                target,
                "rule-1-dual",
                f"'{memory_class.name}' is {occupancy:.1%} occupied ({working_set.bits} of "
                f"{memory_class.size_bits} bits): with {SHRINK_HEADROOM:.0%} of margin the working set already "
                f"needs all {SHRINK_BANKS} of its banks, so there is no smaller size that still holds it",
            )
        )
        return []

    baseline_area = _bundle_area(bundle)
    out: list[Offer] = []
    for banks in range(smallest, min(smallest + SHRINK_STEPS_OFFERED, SHRINK_BANKS)):
        new_size = bank_bits * banks
        # An ABSOLUTE size, not a ratio. A candidate applies its operator to the run's baseline
        # bundle, while the offer was computed on the bundle the evidence came from; those are the
        # same design only in the first wave. "keep 4 of 8 banks" therefore names a different
        # capacity in each of them, and the edit that ran would not be the edit that was priced.
        args = {"cores": list(memory_class.owner_cores), "memory": memory_class.name, "to_bits": new_size}
        cost = _price("core.memory.shrink", args, bundle, budget, target, vetoes)
        if cost is None or baseline_area is None:
            continue
        saved = baseline_area - float(cost["area_mm2"])
        if saved <= 0:
            continue
        out.append(
            Offer(
                operator_id="core.memory.shrink",
                tier=OperatorTier.CORE,
                kind="hardware",
                target={**target, "banks": banks, "of_banks": SHRINK_BANKS, "to_bits": new_size},
                args=args,
                evidence=(
                    f"'{memory_class.name}' is {memory_class.size_bits} bits on cores "
                    f"{list(memory_class.owner_cores)} and the solved placement never puts more than "
                    f"{working_set.bits} bits in it ({occupancy:.1%} occupied). Measured by: "
                    f"{'; '.join(working_set.sources)}"
                ),
                effect=(
                    f"keep {banks} of {memory_class.name}'s {SHRINK_BANKS} banks on cores "
                    f"{list(memory_class.owner_cores)} ({memory_class.size_bits} -> {new_size} bits), "
                    "carrying every aliased view"
                ),
                couples=OPERATORS["core.memory.shrink"].couples,
                predicted_delta=PredictedDelta(
                    value=saved,
                    unit="mm2",
                    scope="bundle",
                    derivation=(
                        f"whole-bundle die area falls from {baseline_area:.3f} to {float(cost['area_mm2']):.3f} mm2 "
                        f"— arithmetic over the cost model, not a solver estimate, so it is exact for the model "
                        f"rather than a bound on it. It buys ZERO cycles: capacity above the working set does no "
                        f"work. The risk it carries is the other way round — the placement and the temporal "
                        f"mapping are re-derived on {new_size} bits and may need more than the "
                        f"{working_set.bits} bits measured here, which the post-hoc check and the latency ceiling "
                        "are what catch."
                    ),
                ),
                cost=cost,
                refs=(
                    fact_ref(ALLOCATION_ARTIFACT, f"{_GROUP}/memory_occupancy"),
                    bundle_ref(f"declared size of '{memory_class.name}' on cores {list(memory_class.owner_cores)}"),
                ),
            )
        )
    return out


def _narrow_offers(  # noqa: PLR0913 -- one call site; each argument is an input to the guard
    evidence: RunEvidence,
    memory_class: _MemoryClass,
    target: dict[str, Any],
    bundle: HardwareBundle,
    budget: HardwareBudget | None,
    vetoes: list[Veto],
) -> list[Offer]:
    """Port-width reduction, selected from measured slack AND measured port occupancy.

    Exactly the two halves Rule 1 uses in the other direction. Slack alone is not enough: it says the
    level kept up, not by how much. Occupancy is the by-how-much, and without it there is no way to
    say which narrowing still keeps up.
    """
    levels = _levels_for(evidence, memory_class)
    if not levels:
        return []
    if any(level.stall_cycles > 0 for level in levels):
        stalling = next(level for level in levels if level.stall_cycles > 0)
        vetoes.append(
            Veto(
                "core.memory.narrow",
                target,
                "rule-1-dual",
                f"'{memory_class.name}' stalls {stalling.stall_cycles:.0f} cycles: it is not keeping up with "
                "the port width it already has, so a narrower one can only cost latency",
            )
        )
        return []
    occupancies = [level.utilization for level in levels if level.utilization is not None]
    if not occupancies:
        vetoes.append(
            Veto(
                "core.memory.narrow",
                target,
                "rule-1-dual",
                f"'{memory_class.name}' has slack but no measured port occupancy, so nothing says HOW MUCH "
                "width is spare; slack alone cannot choose a narrowing factor",
            )
        )
        return []
    occupancy = max(occupancies)
    baseline_area = _bundle_area(bundle)
    out: list[Offer] = []
    for divisor in NARROW_FACTORS:
        projected = occupancy * divisor
        if projected > PORT_OCCUPANCY_CEILING:
            vetoes.append(
                Veto(
                    "core.memory.narrow",
                    {**target, "divisor": divisor},
                    "rule-1-dual",
                    f"the busiest port of '{memory_class.name}' is {occupancy:.3g} occupied, so a {divisor}x "
                    f"narrower one would run at {projected:.3g} against the {PORT_OCCUPANCY_CEILING} ceiling",
                )
            )
            continue
        # Absolute, for the same reason the capacity reduction is: see `_shrink_offers`.
        width = max(1, _widest_port(bundle, memory_class) // divisor)
        args = {"cores": list(memory_class.owner_cores), "memory": memory_class.name, "to_bandwidth": width}
        cost = _price("core.memory.narrow", args, bundle, budget, target, vetoes)
        if cost is None or baseline_area is None:
            continue
        saved = baseline_area - float(cost["area_mm2"])
        if saved <= 0:
            # Narrowing a flop-based register file changes no column IO, so it saves nothing. An
            # offer with a zero delta is noise on the menu, not a move.
            continue
        out.append(
            Offer(
                operator_id="core.memory.narrow",
                tier=OperatorTier.CORE,
                kind="hardware",
                target={**target, "divisor": divisor},
                args=args,
                evidence=(
                    f"'{memory_class.name}' does not stall (slack "
                    f"{max(level.slack_cycles for level in levels):.0f} cycles) and its busiest port is only "
                    f"{occupancy:.3g} occupied over the node's computation span"
                ),
                effect=(
                    f"divide every port bandwidth of '{memory_class.name}' on cores "
                    f"{list(memory_class.owner_cores)} by {divisor}, never below its declared bandwidth_min"
                ),
                couples=OPERATORS["core.memory.narrow"].couples,
                predicted_delta=PredictedDelta(
                    value=saved,
                    unit="mm2",
                    scope="bundle",
                    derivation=(
                        f"the column IO stack is replicated per accessed bit per port, so {divisor}x less width "
                        f"is {divisor}x less of it: {baseline_area:.3f} -> {float(cost['area_mm2']):.3f} mm2. It "
                        f"buys zero cycles as long as the port stays under {PORT_OCCUPANCY_CEILING} occupied "
                        f"({occupancy:.3g} -> {projected:.3g})."
                    ),
                ),
                cost=cost,
                refs=(
                    fact_ref(PROGRESS_ARTIFACT, f"{CORE_COST_NODES_PATH}/*/memory_levels/{memory_class.name}"),
                    bundle_ref(f"port geometry of '{memory_class.name}' on cores {list(memory_class.owner_cores)}"),
                ),
            )
        )
    return out


def _memory_classes(bundle: HardwareBundle) -> list[_MemoryClass]:
    """Distinct physical memory designs in the bundle, each with the cores that own an instance.

    Two collapses happen here, and both are the difference between a design change and a mutation.

    *Alias closures* collapse to their owner: the MXU's ``operand_buffer``, the VPU's and the VMEM
    core's ``vmem`` are one 128 MiB scratchpad seen three ways, so they are one entry, not three.

    *Identical instances* collapse across cores: four TensorCores each own one of those scratchpads,
    and "shrink the VMEM" is a decision about the design of that scratchpad. Shrinking one of the
    four would produce an asymmetric part nobody asked for, and a saving a quarter the size.
    """
    owner_of: dict[tuple[int, str], tuple[int, str]] = {}
    closure: dict[tuple[int, str], set[tuple[int, str]]] = {}
    for group in _alias_groups(bundle):
        refs = sorted(group)
        owner = refs[0]
        closure[owner] = set(refs)
        for ref in refs:
            owner_of[ref] = owner

    classes: dict[tuple[str, int], _MemoryClass] = {}
    for core_id, core in sorted(bundle.cores.items()):
        for name in _declared_memories(core):
            ref = (int(core_id), name)
            owner = owner_of.get(ref, ref)
            if owner != ref:
                continue  # an aliased view; its owner carries the whole closure
            size = _memory_size_bits(bundle, owner)
            if not size:
                continue
            key = (owner[1], size)
            current = classes.get(key)
            members = tuple(sorted(closure.get(owner, {owner})))
            classes[key] = _MemoryClass(
                name=owner[1],
                size_bits=size,
                owner_cores=(current.owner_cores if current else ()) + (owner[0],),
                members=(current.members if current else ()) + members,
            )
    return [classes[key] for key in sorted(classes)]


def _declared_memories(core: dict[str, Any]) -> list[str]:
    """Memory names a core declares, in declaration order (innermost first for a ZigZag core).

    An aie2 tile declares a single unnamed ``memory:`` block; ``"memory"`` is the name the cost model
    already uses for it, so the two agree on what they are pricing and editing.
    """
    if "memories" in core:
        return list((core.get("memories") or {}).keys())
    return ["memory"] if isinstance(core.get("memory"), dict) else []


def _memory_declaration(bundle: HardwareBundle, ref: tuple[int, str]) -> dict[str, Any] | None:
    core = bundle.cores.get(int(ref[0]))
    if core is None:
        return None
    if "memories" in core:
        return ((core.get("memories") or {}).get(ref[1])) or None
    return core.get("memory") if ref[1] == "memory" and isinstance(core.get("memory"), dict) else None


def _memory_size_bits(bundle: HardwareBundle, ref: tuple[int, str]) -> int:
    """Declared capacity in bits. ZigZag calls it ``size``, the aie2 schema calls it ``capacity``."""
    mem = _memory_declaration(bundle, ref) or {}
    return int(mem.get("size") or mem.get("capacity") or 0)


def _top_capacity_memory(core: dict[str, Any]) -> str | None:
    """The memory ``Core.get_memory_capacity()`` reports — the one the allocator's residency is
    measured against.

    ZigZag returns the *top* instance holding the ``I1`` operand, and a hierarchy is declared
    innermost first, so that is the last declared level serving I1. An aie2 tile has exactly one.
    """
    if "memories" not in core:
        return "memory" if isinstance(core.get("memory"), dict) else None
    top = None
    for name, mem in (core.get("memories") or {}).items():
        if "I1" in {str(operand) for operand in mem.get("operands") or ()}:
            top = name
    return top


def _measured_working_set(evidence: RunEvidence, bundle: HardwareBundle, memory_class: _MemoryClass) -> _WorkingSet:
    """The largest occupancy anything measured for this memory, and what measured it.

    Two independent measurements, and the floor is the maximum of them because each is binding in
    its own right: the allocator's residency is what the fused group places in this memory across
    cores, and ZigZag's capacity utilization is what one node's intra-core temporal mapping needs.
    A capacity that clears one and not the other does not hold the design.
    """
    bits = 0
    sources: list[str] = []

    for core_id, name in memory_class.members:
        core = bundle.cores.get(core_id) or {}
        if _top_capacity_memory(core) != name:
            continue  # the allocator constrains a core against its TOP level only
        occupancy = evidence.occupancy_of(core_id)
        if occupancy is None:
            continue
        sources.append(
            f"allocator residency on core {core_id}: {occupancy.resident_bits} of {occupancy.capacity_bits} bits"
        )
        bits = max(bits, occupancy.resident_bits)

    for level, node in _levels_for(evidence, memory_class, with_nodes=True):
        utilization = node.capacity_utilization(level)
        if utilization is None:
            continue
        sources.append(f"CME capacity utilization of '{level.name}' on node '{node.name}': {utilization:.3g}")
        bits = max(bits, int(math.ceil(utilization * memory_class.size_bits)))

    return _WorkingSet(bits=bits, sources=tuple(dict.fromkeys(sources)))


def _levels_for(evidence: RunEvidence, memory_class: _MemoryClass, *, with_nodes: bool = False):
    """The CME memory levels that describe this memory class, optionally paired with their node.

    A level is matched by name on a node that runs on one of the class's cores. ZigZag disambiguates
    two levels backed by one instance as ``name#index``, so the join strips that suffix rather than
    missing the second one.
    """
    cores = memory_class.member_cores
    out = []
    for node in evidence.nodes:
        if not node.modelled or not (set(node.core_ids) & cores):
            continue
        assert node.memory_levels is not None
        for level in node.memory_levels.values():
            if level.name.split("#", 1)[0] != memory_class.name:
                continue
            out.append((level, node) if with_nodes else level)
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
                Veto(
                    "core.array.resize",
                    target,
                    "rule-1",
                    f"node '{node.name}' has evidence=none: nothing measured",
                    unknown=True,
                    refs=(node_ref(node.name),),
                )
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
        # A full array is necessary but not sufficient: the array only sets the latency when the
        # node is not spending its cycles waiting on memory.
        if node.compute_efficiency is not None and node.compute_efficiency < COMPUTE_DOMINATED:
            vetoes.append(
                Veto(
                    "core.array.resize",
                    target,
                    "rule-1-exception",
                    f"node '{node.name}' fills its array but runs at compute_efficiency="
                    f"{node.compute_efficiency:.4g}: its latency is set by memory stalls, not by array "
                    "size, so a larger array would shorten a term that is not binding",
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
                        value=node.ideal_cycles * (1.0 - 1.0 / factor),
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
                    refs=(
                        node_performance_ref(node.name),
                        node_ref(node.name),
                        bundle_ref(f"operational_array of core {core_ids[0]}"),
                    ),
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
                        value=transfer,
                        scope="iteration",
                        derivation=(
                            "upper bound = the transfer-bound cycles of one iteration "
                            f"({transfer:.0f}), i.e. what a tile that overlapped every transfer with "
                            "compute would reclaim. A tile change can also move cycles the other way."
                        ),
                    ),
                    refs=(
                        fact_ref(ALLOCATION_ARTIFACT, f"{OVERLAP_PATH}/recurrence_bound_cycles"),
                        fact_ref(ALLOCATION_ARTIFACT, f"{_GROUP}/performance/bottleneck/transfer_bound_pct"),
                        per_iteration_ref(),
                        declared_ref("intra_core_tiling", "launch parameter"),
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
                    value=0.0,
                    scope="iteration",
                    derivation=(
                        "an infeasible solve has no latency to improve on; cutting the group removes "
                        f"the {unmet.gap:.6g} {unmet.unit} of residency it cannot afford. The cost is "
                        "off-chip traffic for the intermediate that stops being fused."
                    ),
                ),
                refs=(
                    fact_ref(INFEASIBILITY_ARTIFACT, "/resources/*/unmet"),
                    fact_ref(ALLOCATION_ARTIFACT, f"{_GROUP}/fused_groups"),
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
                value=transfer,
                scope="iteration",
                derivation=(
                    f"upper bound = the transfer-bound cycles of one iteration ({transfer:.0f}); fusing "
                    "can at best keep every intermediate on chip and hide the rest behind compute. "
                    "Re-fusing also raises on-chip pressure and can make a group infeasible."
                ),
            ),
            refs=(
                fact_ref(ALLOCATION_ARTIFACT, f"{_GROUP}/performance/bottleneck/transfer_bound_pct"),
                fact_ref(ALLOCATION_ARTIFACT, f"{_GROUP}/fused_groups"),
                per_iteration_ref(),
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
    refs: tuple[dict[str, str], ...],
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
        refs=refs,
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
                value=peak - mean,
                scope="iteration",
                derivation=(
                    f"busy_i = latency_per_iteration - slack_i; peak {peak:.0f}, mean {mean:.0f}. No "
                    "redistribution can go below the mean, so the ceiling is their difference. It is "
                    "only reachable if the added cores can actually run the binding resource's "
                    "operators -- operator_types may forbid it, in which case the saving is zero."
                ),
            ),
            refs=(binding_ref(), slack_ref(), per_iteration_ref(), declared_ref("nb_cols_to_use", "launch parameter")),
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
                        value=saving,
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
                    refs=(
                        binding_ref(),
                        slack_ref(),
                        per_iteration_ref(),
                        bundle_ref(f"core_connectivity link at {args['link_bandwidth']} bits/cycle"),
                    ),
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
    elif operator_id == "core.memory.shrink":
        notes += _scale_memory(edited, args["cores"], args["memory"], capacity_bits=int(args["to_bits"]))
    elif operator_id == "core.memory.narrow":
        notes += _scale_memory(edited, args["cores"], args["memory"], bandwidth_bits=int(args["to_bandwidth"]))
    elif operator_id == "core.array.resize":
        notes += _resize_array(edited, args["cores"], {str(d): int(f) for d, f in args["dims"].items()})
    elif operator_id == "link.bandwidth":
        notes += _scale_links(edited, int(args["link_bandwidth"]), int(args["factor"]))
    return AppliedOperator(operator_id, edited, params, notes)


def _scale_memory(  # noqa: PLR0913 -- four independent knobs on one edit; splitting duplicates the walk
    bundle: HardwareBundle,
    core_ids: list[int],
    memory: str,
    *,
    capacity: int = 1,
    bandwidth: int = 1,
    capacity_bits: int | None = None,
    bandwidth_bits: int | None = None,
    follow_aliases: bool = True,
) -> list[str]:
    """Scale one memory level's capacity and/or port widths on the given cores, either way.

    Aliased views are carried along by default. The MXU's ``operand_buffer``, the VPU's and the VMEM
    core's ``vmem`` are three views of one 128 MiB scratchpad; widening only the view the stalling
    node happens to sit behind would give the solve a wider port that the cost model never bills and
    the other views never see -- a mutation that is free precisely because it is incoherent. A
    *shrink* has the sharper version of the same problem: leaving one view at 128 MiB would let the
    solver keep placing tensors the shrunken silicon cannot hold.
    """
    targets = {(int(c), memory) for c in core_ids}
    if follow_aliases:
        for group in _alias_groups(bundle):
            if targets & group:
                targets |= group
    notes: list[str] = []
    for core_id, mem_name in sorted(targets):
        mem = _memory_declaration(bundle, (core_id, mem_name))
        if mem is None:
            continue
        # ZigZag declares capacity as `size`, the aie2 schema as `capacity`. Edit whichever is there
        # rather than introducing the other key, which the validator would reject.
        size_key = "size" if "size" in mem else ("capacity" if "capacity" in mem else None)
        if size_key and capacity > 1:
            mem[size_key] = int(mem.get(size_key, 0)) * capacity
            notes.append(f"core {core_id}: {mem_name}.{size_key} x{capacity} -> {mem[size_key]}")
        if size_key and capacity_bits is not None:
            # `min`, so a reduction can only ever reduce: applied to a bundle already smaller than
            # the offer's target this would otherwise be a silent GROWTH, and one that no budget
            # guard ever priced because the operator declares itself a reduction.
            mem[size_key] = min(int(mem.get(size_key, 0)) or capacity_bits, capacity_bits)
            notes.append(f"core {core_id}: {mem_name}.{size_key} -> {mem[size_key]}")
        notes += _scale_ports(mem, core_id, mem_name, bandwidth, bandwidth_bits)
    return notes


def _scale_ports(mem: dict[str, Any], core_id: int, mem_name: str, factor: int, to_bandwidth: int | None) -> list[str]:
    """Widen every port of one memory declaration by `factor`, or cap them all at `to_bandwidth`.

    A narrowing never drops a port below its declared ``bandwidth_min``: that is the smallest access
    the memory supports, and a maximum below it would describe hardware that cannot service its own
    minimum request. It is a cap rather than a set, so applying it to a bundle already narrower than
    the target cannot widen it. An aie2 tile carries the pair on the memory itself, not on ports.
    """
    if factor <= 1 and to_bandwidth is None:
        return []
    ports = mem.get("ports")
    declarations = list(ports) if ports else [mem]
    floor = min((int(d["bandwidth_min"]) for d in declarations if "bandwidth_min" in d), default=1)
    for declaration in declarations:
        for key in ("bandwidth_min", "bandwidth_max"):
            if key not in declaration:
                continue
            if factor > 1:
                declaration[key] = int(declaration[key]) * factor
            if to_bandwidth is not None:
                declaration[key] = max(floor, min(int(declaration[key]), to_bandwidth))
    verb = f"x{factor}" if factor > 1 else f"<= {to_bandwidth}"
    return [f"core {core_id}: {mem_name} port bandwidths {verb}"]


def _widest_port(bundle: HardwareBundle, memory_class: _MemoryClass) -> int:
    """The widest declared access of this memory, which a narrowing divides."""
    mem = _memory_declaration(bundle, (memory_class.owner_cores[0], memory_class.name)) or {}
    ports = mem.get("ports") or [mem]
    return max((int(p.get("bandwidth_max", 0)) for p in ports), default=0)


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


# ── The post-hoc guards ─────────────────────────────────────────────────────────────────────────

REDUCTION_OPERATORS = frozenset({"core.memory.shrink", "core.memory.narrow"})
"""Operators that make hardware smaller. They share a post-hoc guard the growths do not need."""


def post_hoc_reduction_check(
    operator_id: str,
    target: dict[str, Any],
    after: RunEvidence,
    *,
    parent_latency_cycles: float | None = None,
    objective: Objective | None = None,
) -> str | None:
    """Verify after the solve that a reduction did not break what it made smaller. None = accepted.

    The pre-solve guard is arithmetic over the *previous* run's placement, and that placement is
    re-derived on the smaller hardware: LOMA picks a fresh temporal mapping and the allocator a
    fresh placement, either of which may need more than the old one did. So the reduction is only
    banked once the run that followed it says so.

    Two failures, and neither is a smaller design:

    * **infeasible** — the cut crossed the real working set. That is a hard fact about this
      (hardware, workload) pair and worth caching: no capacity at or below this one holds it.
    * **slower than the objective allows** — the mapping still fits but pays for it in cycles.
      Trading latency for area is exactly what the ``area`` objective's latency ceiling exists to
      bound, so a candidate over that ceiling is rejected rather than reported as an area win.
    """
    if operator_id not in REDUCTION_OPERATORS:
        return None
    detail = f"'{target.get('memory')}'" + (f" -> {target['to_bits']} bits" if target.get("to_bits") else "")
    if not after.feasible:
        return (
            f"the reduction of {detail} made the mapping INFEASIBLE: the pre-solve floor came from the "
            "previous placement, and the placement re-derived on the smaller hardware needs more. No "
            "capacity at or below this one holds this workload"
        )
    if after.latency_total is None:
        return f"the run after reducing {detail} reported no latency, so the reduction cannot be verified"
    if objective is not None:
        violations = objective.violations(after.latency_total, None)
        if violations:
            return f"the reduction of {detail} is within the working set but {'; '.join(violations)}"
    if parent_latency_cycles is not None and after.latency_total > parent_latency_cycles:
        regression = after.latency_total - parent_latency_cycles
        floor = after.noise_floor
        if floor.clears(regression):
            return (
                f"the reduction of {detail} cost {regression:.0f} cycles "
                f"({parent_latency_cycles:.0f} -> {after.latency_total:.0f}), above the noise floor of "
                f"{floor.cycles:.0f} ({floor.source}): the smaller memory forced a worse temporal mapping"
            )
    return None


def post_hoc_check(operator_id: str, target: dict[str, Any], before: RunEvidence, after: RunEvidence) -> str | None:
    """Verify after the solve that a coupled resize or a reduction actually held. None = accepted."""
    if operator_id in REDUCTION_OPERATORS:
        return post_hoc_reduction_check(operator_id, target, after, parent_latency_cycles=before.latency_total)
    if operator_id != "core.array.resize":
        return None
    name = str(target.get("node"))
    old, new = before.node(name), after.node(name)
    if old is None or new is None:
        return None
    return post_hoc_utilization_check(name, old.mac_spatial_utilization, new.mac_spatial_utilization)


def post_hoc_utilization_check(node: str, before: float | None, after: float | None) -> str | None:
    """The scalar form of :func:`post_hoc_check`, for a caller that carries the parent's number
    rather than the parent's whole evidence. None = accepted.

    ZigZag's validator checks none of the coupling and degrades silently —
    ``limit_unrolling_to_mem_bandwidth`` merely logs when the unrolling it was asked for does not fit
    the memory bandwidth. So the only reliable test that the coupling was right is the outcome: if
    the node's spatial utilization fell, the larger array is not being fed and the mutation must be
    rejected rather than scored on latency it did not earn.
    """
    if before is None or after is None:
        return (
            f"node '{node}' reports no mac_spatial_utilization on both sides of the resize, so the "
            "coupling cannot be verified; rejecting rather than crediting an unverified change"
        )
    if after < before:
        return (
            f"node '{node}' mac_spatial_utilization fell {before:.4g} -> {after:.4g}: the resized "
            "array is not being fed, so the coupled memory rescale was wrong for this core"
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
        **price_of(verdict.report),
        "budget": budget.label if budget else None,
        "max_area_mm2": budget.max_area_mm2 if budget else None,
    }


def price_of(report) -> dict[str, Any]:
    """The three budgetable scalars *and the caveats the model publishes about them*.

    Three numbers used to be all that survived here. The model states its own tolerance under a
    heading called ACCURACY CLAIM, warns when it priced a bundle at a technology node the bundle
    never declared, counts the memories whose authored energy it disagrees with by more than 2x, and
    records when a core's compute area was not modelled at all -- and every one of those was thrown
    away one line after being computed, leaving a bare `area_mm2` that reads as exact.

    Quoting a producer's own published tolerance is not inventing one. Nothing here is modelled or
    estimated; it is the report, un-discarded.
    """
    return {
        "area_mm2": report.total_area_mm2,
        "peak_access_energy_pj_per_cycle": report.peak_access_energy_pj_per_cycle,
        "technology_node": report.technology_node,
        # False = priced at the default node because the bundle named none. The one documented
        # substitution in this number, and the thing that flips how honest it is.
        "technology_declared": report.technology_declared,
        # False = at least one on-die core declares no array, so the area is a LOWER BOUND.
        "compute_modelled": report.compute_modelled,
        "accuracy_claim": report.accuracy_claim,
        "authored_disagreements": list(report.authored_disagreements),
        "warnings": list(report.warnings),
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
