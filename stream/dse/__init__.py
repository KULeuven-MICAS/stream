"""Guarded design-space-exploration operators.

A DSE agent that may edit anything will edit the things that do not matter. This package encodes
what is *legal* to change as executable preconditions over measured evidence, so an operator that
no evidence supports is never offered at all — the judgement left to a human or an LLM is
*which* of the legal moves to take and why, not what the numbers are.

Two modules:

* :mod:`stream.dse.evidence` — the typed reading of one run: per-(node, level) stalls with an
  explicit "was anything modelled here" flag, per-node compute efficiency, the solver's overlap
  and binding set, the II decomposition, the optimality gap, and the infeasibility diagnosis.
* :mod:`stream.dse.operators` — the registry: each operator declares a precondition over that
  evidence, the concrete edit it makes, the invariants that must move with it, and a predicted
  cycle saving with its derivation. Hardware edits are priced against a budget *before* any solve.
"""

from stream.dse.evidence import (
    MemoryLevelEvidence,
    NodeEvidence,
    NoiseFloor,
    RunEvidence,
)
from stream.dse.operators import (
    OFFER_TIERS,
    OPERATORS,
    AppliedOperator,
    Offer,
    Operator,
    OperatorTier,
    PredictedDelta,
    Veto,
    apply_operator,
    offer_operators,
    post_hoc_check,
    post_hoc_utilization_check,
)

__all__ = [
    "OFFER_TIERS",
    "OPERATORS",
    "AppliedOperator",
    "MemoryLevelEvidence",
    "NodeEvidence",
    "NoiseFloor",
    "Offer",
    "Operator",
    "OperatorTier",
    "PredictedDelta",
    "RunEvidence",
    "Veto",
    "apply_operator",
    "offer_operators",
    "post_hoc_check",
    "post_hoc_utilization_check",
]
