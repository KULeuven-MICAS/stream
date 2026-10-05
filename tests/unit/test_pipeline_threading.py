"""The solve options reach the allocation stage through the context, with their defaults set by the api alone."""

from unittest.mock import MagicMock

import pytest

from stream.api import DEFAULT_TIME_LIMIT_S, SolveOptions
from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES, drop_families, load_families
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


def test_stage_reads_solve_options_from_context():
    families = load_families(drop_families(DEFAULT_FAMILIES, ["dma_channels"]))
    ctx = StageContext.from_kwargs(
        allocation_problem=MagicMock(),
        output_path="/tmp/test",
        backend="ORTOOLS_GSCIP",
        families=families,
        time_limit_s=12.5,
        solver_log=True,
        artifacts=False,
    )
    stage = AllocationStage([MagicMock()], ctx)
    assert stage.families is families
    assert (stage.backend, stage.time_limit_s, stage.solver_log, stage.artifacts) == (
        "ORTOOLS_GSCIP",
        12.5,
        True,
        False,
    )


def test_stage_has_no_defaults_of_its_own():
    ctx = StageContext.from_kwargs(allocation_problem=MagicMock(), output_path="/tmp/test", backend="ORTOOLS_GSCIP")
    with pytest.raises(StageContractError, match="families"):
        AllocationStage([MagicMock()], ctx)
