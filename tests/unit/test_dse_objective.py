"""What "better" means, and what happens when a prediction is wrong.

Two things are pinned here, and both are the kind that fail silently if nobody checks them:

* an objective whose ceilings are advisory rather than enforced -- a search that reports an
  over-budget design as its best result and nobody notices, because "best" was a bare comparison;
* a predicted delta nobody scores, which is a guess with extra steps. The live case is a run that
  predicted a 9,088-cycle saving and came back 24,178 cycles slower.
"""

import math

import pytest

from stream.dse.objective import (
    DEFAULT_LATENCY_TOLERANCE,
    Objective,
    ObjectiveKind,
)
from stream.dse.residual import MIN_TRUST, OperatorScorecard, Residual

BASELINE_LATENCY = 1_392_195.0
BASELINE_AREA = 172.78


class TestObjective:
    def test_the_default_is_latency_under_an_area_ceiling(self):
        """The historical behaviour, unchanged: minimise cycles, never above the baseline's silicon."""
        objective = Objective.from_baseline(baseline_latency_cycles=BASELINE_LATENCY, baseline_area_mm2=BASELINE_AREA)
        assert objective.kind is ObjectiveKind.LATENCY
        assert objective.max_area_mm2 == pytest.approx(BASELINE_AREA)
        # No latency ceiling: latency IS the thing being minimised, and a ceiling on the objective
        # itself would only forbid candidates that are worse anyway.
        assert objective.max_latency_cycles is None
        assert objective.value(BASELINE_LATENCY / 2, BASELINE_AREA) == BASELINE_LATENCY / 2

    def test_an_area_objective_always_carries_a_latency_ceiling(self):
        """Unconstrained "minimise area" is won by a design too small to run the workload. The
        ceiling is what makes the answer an engineering result instead of a degenerate one."""
        objective = Objective.from_baseline(
            "area", baseline_latency_cycles=BASELINE_LATENCY, baseline_area_mm2=BASELINE_AREA
        )
        assert objective.max_latency_cycles == pytest.approx(BASELINE_LATENCY * (1 + DEFAULT_LATENCY_TOLERANCE))
        assert objective.value(BASELINE_LATENCY, 51.7) == 51.7
        # Half the silicon for twice the time is not an area win, it is a different design.
        assert objective.value(BASELINE_LATENCY * 2, 51.7) == math.inf

    def test_a_budget_violation_is_infinite_not_merely_penalised(self):
        """Enforced by construction. A guard that each call site has to remember is a guard that
        will eventually be forgotten, and the forgetting is silent."""
        objective = Objective.from_baseline(baseline_latency_cycles=BASELINE_LATENCY, baseline_area_mm2=BASELINE_AREA)
        assert objective.value(1.0, BASELINE_AREA * 1.01) == math.inf
        assert objective.violations(1.0, BASELINE_AREA * 1.01)

    def test_a_missing_term_scores_infinite_rather_than_partially(self):
        """Ranking an unpriced candidate against a priced one is how an unbudgeted design wins a
        budgeted comparison."""
        objective = Objective.from_baseline("efficiency", baseline_latency_cycles=BASELINE_LATENCY)
        assert objective.value(BASELINE_LATENCY, None) == math.inf
        assert objective.value(None, 51.7) == math.inf

    def test_efficiency_is_the_area_delay_product_and_is_scale_free(self):
        """k times the silicon for k times the speed leaves the score alone, which is the trade a
        co-design search should be neutral about."""
        objective = Objective(kind=ObjectiveKind.EFFICIENCY)
        assert objective.value(1000.0, 100.0) == objective.value(500.0, 200.0)
        assert objective.value(500.0, 100.0) < objective.value(1000.0, 100.0)
        assert objective.unit == "cycles*mm2"

    def test_the_serialized_objective_names_its_ceilings(self):
        """A comparison record that cannot see the objective silently mixes two different questions."""
        payload = Objective.from_baseline(
            "area", baseline_latency_cycles=BASELINE_LATENCY, baseline_area_mm2=BASELINE_AREA
        ).as_dict()
        assert payload["kind"] == "area"
        assert payload["unit"] == "mm2"
        assert payload["baseline_value"] == pytest.approx(BASELINE_AREA)
        assert payload["max_latency_cycles"] == pytest.approx(BASELINE_LATENCY * 1.02)


class TestResidual:
    def test_the_summary_is_predicted_achieved_unexplained(self):
        """The exact line the stop-on-bound report has to be able to print."""
        residual = Residual("system.tiling.intra_core", predicted=9088.0, achieved=-24178.0, unit="cycles")
        assert residual.residual == pytest.approx(33266.0)
        assert "predicted 9088" in residual.summary()
        assert "unexplained 3.327e+04" in residual.summary() or "unexplained 33266" in residual.summary()

    def test_an_untried_operator_is_trusted(self):
        """Untried is not unreliable. Starting at zero would mean an operator can never earn a first
        application, which quietly shrinks the action space to whatever ran first."""
        assert OperatorScorecard().trust("core.memory.bandwidth") == 1.0

    def test_conservative_predictions_are_not_punished(self):
        """An upper bound is SUPPOSED to over-shoot the other way. Punishing a delivered 2,000 for a
        promised 463 would push the registry towards optimistic bounds."""
        scorecard = OperatorScorecard()
        scorecard.record(Residual("core.memory.bandwidth", predicted=463.0, achieved=2000.0, unit="cycles"))
        assert scorecard.trust("core.memory.bandwidth") == 1.0

    def test_a_move_that_made_things_worse_falls_to_the_floor(self):
        """The live TPU7x case. A sign error is not a calibration error."""
        scorecard = OperatorScorecard()
        scorecard.record(Residual("system.tiling.intra_core", predicted=9088.0, achieved=-24178.0, unit="cycles"))
        assert scorecard.trust("system.tiling.intra_core") == pytest.approx(MIN_TRUST)
        assert scorecard.discounted("system.tiling.intra_core", 9088.0) == pytest.approx(9088.0 * MIN_TRUST)

    def test_repeated_over_prediction_compounds_geometrically(self):
        """The quantity being combined is a ratio: 2x then 8x is 4x, not 5x."""
        scorecard = OperatorScorecard()
        scorecard.record(Residual("op", predicted=200.0, achieved=100.0, unit="cycles"))
        scorecard.record(Residual("op", predicted=800.0, achieved=100.0, unit="cycles"))
        # Mean over-prediction 4x; with a half-life of 2x that is two halvings.
        assert scorecard.trust("op") == pytest.approx(0.25)

    def test_an_operator_is_deprioritised_never_removed(self):
        """It stays in the menu because the NEXT evidence may be the case it is right about, and a
        registry that deleted operators would silently shrink its own action space."""
        scorecard = OperatorScorecard()
        for _ in range(10):
            scorecard.record(Residual("op", predicted=1e9, achieved=-1.0, unit="cycles"))
        assert scorecard.trust("op") == MIN_TRUST > 0

    def test_an_unmeasured_outcome_does_not_score(self):
        """A candidate that never came back is not evidence the operator over-predicted."""
        scorecard = OperatorScorecard()
        scorecard.record(Residual("op", predicted=500.0, achieved=None, unit="cycles"))
        assert scorecard.trust("op") == 1.0
        assert scorecard.as_dict()["history"][0]["residual"] is None
