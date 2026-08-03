"""Predicted vs achieved, scored — what makes the loop agentic rather than generative.

Every offer carries a ``predicted_delta`` with a derivation. That is only worth something if
somebody checks it. This module is the check: for each applied operator it records what was
predicted, what the next solve actually delivered, and the gap between them; over several
applications it turns that history into a **trust factor** that reorders the menu.

THE LIVE EXAMPLE THIS EXISTS FOR
--------------------------------
On TPU7x SwiGLU seq=2048 an operator predicted a 9,088-cycle saving and the run came back
**24,178 cycles slower** — a residual of 33,266 cycles, and a sign error, not a magnitude error.
Nothing in the loop noticed, so the same family of move stayed at the top of the menu.

WHAT IS AND IS NOT PENALISED
----------------------------
Only *over-prediction* is penalised. An operator that promised 463 cycles and delivered 2,000 was
conservative, which is what an upper bound is supposed to be; punishing that would push the
registry towards optimistic bounds, which is the opposite of what the guards are for.

The penalty is a ratio, not a difference, so it is comparable across an operator predicting
hundreds of cycles and one predicting mm² of silicon. An operator with no history is trusted at
1.0 — untried is not the same as unreliable, and starting it at zero would mean an operator can
never earn a first application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MIN_TRUST = 0.1
"""Floor on the trust factor. An operator that has always over-predicted is deprioritised, never
removed: it stays in the menu because the *next* evidence may be the case it is right about, and a
registry that deleted operators would silently shrink its own action space."""

TRUST_HALF_LIFE = 2.0
"""Over-predicting by this ratio (predicted / achieved) halves the trust factor.

2.0 = "promised twice what it delivered". Chosen so the 9,088-predicted / -24,178-achieved case
lands at the floor rather than merely nudged: a sign error is not a calibration error.
"""


@dataclass(frozen=True)
class Residual:
    """One applied operator's prediction against its outcome.

    ``achieved`` and ``predicted`` are both *improvements* in the objective's unit and are positive
    when the move helped. A negative ``achieved`` therefore means the operator made things worse,
    which is the case the ratio below has to handle explicitly rather than by dividing by it.
    """

    operator_id: str
    predicted: float
    achieved: float | None
    unit: str
    objective: str = "latency"
    note: str = ""

    @property
    def residual(self) -> float | None:
        """predicted - achieved: what the prediction failed to explain. Positive = over-predicted."""
        return None if self.achieved is None else self.predicted - self.achieved

    @property
    def over_prediction_ratio(self) -> float | None:
        """How many times more the operator promised than it delivered. None when unmeasured.

        A move that made the objective *worse* (``achieved <= 0``) has no finite ratio to report,
        so it is reported as infinite over-prediction — which is what it is, and what drives the
        trust factor to its floor.
        """
        if self.achieved is None or self.predicted <= 0:
            return None
        if self.achieved <= 0:
            return math.inf
        return self.predicted / self.achieved

    def as_dict(self) -> dict[str, Any]:
        ratio = self.over_prediction_ratio
        return {
            "operator": self.operator_id,
            "objective": self.objective,
            "unit": self.unit,
            "predicted": self.predicted,
            "achieved": self.achieved,
            "residual": self.residual,
            "over_prediction_ratio": None if ratio is None or math.isinf(ratio) else ratio,
            "made_it_worse": self.achieved is not None and self.achieved < 0,
            "note": self.note,
        }

    def summary(self) -> str:
        """The one line E5 has to be able to print: predicted X, achieved Y, unexplained Z."""
        if self.achieved is None:
            return f"{self.operator_id}: predicted {self.predicted:.4g} {self.unit}, outcome not measured"
        return (
            f"{self.operator_id}: predicted {self.predicted:.4g} {self.unit}, "
            f"achieved {self.achieved:.4g}, unexplained {self.residual:.4g}"
        )


@dataclass
class OperatorScorecard:
    """Per-operator prediction history and the trust factor it earns."""

    residuals: list[Residual] = field(default_factory=list)

    def record(self, residual: Residual) -> None:
        self.residuals.append(residual)

    def trust(self, operator_id: str) -> float:
        """Multiplier in ``[MIN_TRUST, 1.0]`` for this operator's predicted delta.

        Geometric over the recorded over-predictions, because the quantity being combined is a
        ratio: an operator that over-predicted 2x then 8x should score as 4x, not 5x.
        """
        ratios = [
            r.over_prediction_ratio
            for r in self.residuals
            if r.operator_id == operator_id and r.over_prediction_ratio is not None
        ]
        if not ratios:
            return 1.0  # untried is not unreliable
        if any(math.isinf(ratio) for ratio in ratios):
            return MIN_TRUST
        mean_log = sum(math.log(max(ratio, 1.0)) for ratio in ratios) / len(ratios)
        # trust = 0.5 ** (log2 of the mean over-prediction) => a TRUST_HALF_LIFE-fold miss halves it.
        exponent = mean_log / math.log(TRUST_HALF_LIFE)
        return max(MIN_TRUST, 0.5**exponent)

    def discounted(self, operator_id: str, predicted: float) -> float:
        """`predicted`, scaled by what this operator's history says it is actually worth."""
        return predicted * self.trust(operator_id)

    def as_dict(self) -> dict[str, Any]:
        operators = sorted({r.operator_id for r in self.residuals})
        return {
            "history": [r.as_dict() for r in self.residuals],
            "trust": {op: self.trust(op) for op in operators},
        }

    def report(self) -> str:
        """Human-readable scorecard, worst-trusted first — the text a proposer prompt embeds."""
        operators = sorted({r.operator_id for r in self.residuals}, key=self.trust)
        if not operators:
            return "No operator has been scored yet."
        lines = []
        for op in operators:
            applications = [r for r in self.residuals if r.operator_id == op]
            lines.append(
                f"  - {op}: trust {self.trust(op):.2f} over {len(applications)} application(s); "
                + "; ".join(r.summary().split(": ", 1)[1] for r in applications)
            )
        return "\n".join(lines)
