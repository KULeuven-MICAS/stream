"""What "better" means, made explicit.

Until now a DSE run had exactly one notion of better -- lower latency -- and one budget -- the
baseline's silicon. That is the right default and it is not the only useful question. On a design
already at 98.9% of its own MXU roofline there is ~1% of latency to win and a *large* amount of
silicon to give back, and a search that can only minimise latency cannot even express the second
result, let alone find it.

THREE OBJECTIVES, EACH WITH ITS OWN GUARD
-----------------------------------------
``latency``
    Minimise cycles subject to an area ceiling. The default, and the historical behaviour.
``area``
    Minimise mm² subject to a latency ceiling. **The latency ceiling is mandatory, not optional.**
    Unconstrained "minimise area" is won by a design too small to run the workload at all; the
    ceiling is what makes the answer an engineering result instead of a degenerate one.
``efficiency``
    Minimise the area-delay product, latency x area, with *both* ceilings still in force.

WHY AREA-DELAY PRODUCT FOR ``efficiency``
-----------------------------------------
ADP is scale-free in exactly the trade a co-design search should be neutral about: spending k times
the silicon to go k times faster leaves it unchanged, so it ranks designs by how well they convert
silicon into speed rather than by how much of either they use. Performance-per-mm² is its reciprocal
and induces the identical ordering, so the choice between them is presentational; ADP is used here
because it is a cost (lower is better), which is the same direction as the other two objectives and
therefore needs no sign flip anywhere downstream.

Keeping both ceilings under ``efficiency`` is not redundant. The product alone is happy to trade a
10x area cut for a 9x latency loss; the ceilings are what say that neither of those is a design
anyone asked for.

BUDGETS ARE ENFORCED BY CONSTRUCTION
------------------------------------
:meth:`Objective.value` returns ``inf`` for a candidate outside its budget, so an over-budget
variant can never be selected as best by any comparison anywhere. That is deliberate: a guard that
has to be remembered at each call site is a guard that will eventually be forgotten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DEFAULT_LATENCY_TOLERANCE = 0.02
"""How much slower than the baseline an ``area``/``efficiency`` candidate may be, as a fraction.

2% is deliberately of the same order as the conservative noise floor in :mod:`stream.dse.evidence`:
a latency regression the run cannot distinguish from solver noise is the most a design should be
asked to pay for silicon it does not need.
"""

DEFAULT_AREA_TOLERANCE = 0.0
"""How much more silicon than the baseline a candidate may cost. Zero: equal-or-less by default."""


class ObjectiveKind(StrEnum):
    LATENCY = "latency"
    AREA = "area"
    EFFICIENCY = "efficiency"


@dataclass(frozen=True)
class Objective:
    """A direction to improve in, plus the ceilings that keep the answer honest."""

    kind: ObjectiveKind = ObjectiveKind.LATENCY
    max_area_mm2: float | None = None
    max_latency_cycles: float | None = None
    baseline_latency_cycles: float | None = None
    baseline_area_mm2: float | None = None

    @classmethod
    def from_baseline(
        cls,
        kind: str | ObjectiveKind = ObjectiveKind.LATENCY,
        *,
        baseline_latency_cycles: float | None = None,
        baseline_area_mm2: float | None = None,
        latency_tolerance: float = DEFAULT_LATENCY_TOLERANCE,
        area_tolerance: float = DEFAULT_AREA_TOLERANCE,
    ) -> Objective:
        """Derive both ceilings from the baseline measurement.

        Every objective gets an area ceiling and every objective except ``latency`` gets a latency
        ceiling. ``latency`` deliberately gets none: it is *already* the thing being minimised, and
        a ceiling on the objective itself would only forbid candidates that are worse anyway.
        """
        kind = ObjectiveKind(str(kind))
        max_area = baseline_area_mm2 * (1.0 + area_tolerance) if baseline_area_mm2 else None
        max_latency = (
            baseline_latency_cycles * (1.0 + latency_tolerance)
            if baseline_latency_cycles and kind is not ObjectiveKind.LATENCY
            else None
        )
        return cls(
            kind=kind,
            max_area_mm2=max_area,
            max_latency_cycles=max_latency,
            baseline_latency_cycles=baseline_latency_cycles,
            baseline_area_mm2=baseline_area_mm2,
        )

    # ── Scoring ─────────────────────────────────────────────────────────────────────────────

    def violations(self, latency_cycles: float | None, area_mm2: float | None) -> list[str]:
        """Which ceilings this candidate busts. Empty means it is admissible."""
        out: list[str] = []
        if self.max_area_mm2 is not None and area_mm2 is not None and area_mm2 > self.max_area_mm2:
            out.append(f"area {area_mm2:.3f} mm2 exceeds the {self.max_area_mm2:.3f} mm2 ceiling")
        if (
            self.max_latency_cycles is not None
            and latency_cycles is not None
            and latency_cycles > self.max_latency_cycles
        ):
            out.append(f"latency {latency_cycles:.0f} exceeds the {self.max_latency_cycles:.0f} cycle ceiling")
        return out

    def value(self, latency_cycles: float | None, area_mm2: float | None) -> float:
        """The number to minimise. ``inf`` when a ceiling is busted or a needed term is missing.

        A missing term is ``inf`` rather than a partial score on purpose: ranking an unpriced
        candidate against a priced one is how an unbudgeted design wins a budgeted comparison.
        """
        if self.violations(latency_cycles, area_mm2):
            return math.inf
        match self.kind:
            case ObjectiveKind.LATENCY:
                return latency_cycles if latency_cycles is not None else math.inf
            case ObjectiveKind.AREA:
                return area_mm2 if area_mm2 is not None else math.inf
            case ObjectiveKind.EFFICIENCY:
                if latency_cycles is None or area_mm2 is None:
                    return math.inf
                return latency_cycles * area_mm2

    @property
    def unit(self) -> str:
        return {
            ObjectiveKind.LATENCY: "cycles",
            ObjectiveKind.AREA: "mm2",
            ObjectiveKind.EFFICIENCY: "cycles*mm2",
        }[self.kind]

    @property
    def improves_with(self) -> str:
        """The quantity an operator must move for this objective, for a proposer to read."""
        return {
            ObjectiveKind.LATENCY: "cycles saved",
            ObjectiveKind.AREA: "area saved (mm2)",
            ObjectiveKind.EFFICIENCY: "cycles saved OR area saved (mm2) -- their product is the score",
        }[self.kind]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "unit": self.unit,
            "max_area_mm2": self.max_area_mm2,
            "max_latency_cycles": self.max_latency_cycles,
            "baseline_area_mm2": self.baseline_area_mm2,
            "baseline_latency_cycles": self.baseline_latency_cycles,
            "baseline_value": self.value(self.baseline_latency_cycles, self.baseline_area_mm2),
        }

    def __str__(self) -> str:
        ceilings = []
        if self.max_area_mm2 is not None:
            ceilings.append(f"area <= {self.max_area_mm2:.3f} mm2")
        if self.max_latency_cycles is not None:
            ceilings.append(f"latency <= {self.max_latency_cycles:.0f} cycles")
        return f"minimise {self.kind} [{self.unit}]" + (f" subject to {', '.join(ceilings)}" if ceilings else "")
