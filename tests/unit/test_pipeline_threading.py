"""The solve options reach the allocation stage through the context, with their defaults set by the api alone."""

from unittest.mock import MagicMock

import pytest

from stream.api import DEFAULT_TIME_LIMIT_S, SolveOptions
from stream.stages.allocation.steady_state_allocation import AllocationStage
from stream.stages.context import StageContext, StageContractError


def test_solve_options_carry_families():
    assert SolveOptions().families is None
    assert SolveOptions(families=["placement"]).families == ["placement"]


def test_solve_options_default_the_time_limit_and_reject_a_non_positive_one():
    assert SolveOptions().time_limit_s == DEFAULT_TIME_LIMIT_S
    for limit in (0, -5):
        with pytest.raises(ValueError, match="time_limit_s must be positive"):
            SolveOptions(time_limit_s=limit)


def test_solve_options_take_one_route_by_default_and_reject_fewer_or_a_fraction():
    assert SolveOptions().route_alternatives == 1
    for alternatives in (0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="route_alternatives must be an integer of at least 1"):
            SolveOptions(route_alternatives=alternatives)


def test_solve_options_keep_the_meaning_of_their_positional_arguments():
    assert SolveOptions("gurobi", 2, "even").temporal_mapping_type == "even"


def test_stage_has_no_defaults_of_its_own():
    ctx = StageContext.from_kwargs(allocation_problem=MagicMock(), output_path="/tmp/test", backend="ORTOOLS_GSCIP")
    with pytest.raises(StageContractError, match="families"):
        AllocationStage([MagicMock()], ctx)
