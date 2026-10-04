"""Unit tests for pipeline threading of the solve options into the allocation stage.

Covers:
  - SolveOptions carries constraint_selection, None by default
  - AllocationStage reads constraint_selection, the time limit and the solver log from context
  - AllocationStage defaults to ConstraintSelection(), 300 s and no solver log when absent
"""

from unittest.mock import MagicMock

from stream.opt.solver import ConstraintSelection
from stream.stages.allocation.steady_state_allocation import AllocationStage
from stream.stages.context import StageContext


def test_solve_options_carry_constraint_selection():
    from stream.api import SolveOptions

    assert SolveOptions().constraint_selection is None
    selection = ConstraintSelection(dma_channels=False)
    assert SolveOptions(constraint_selection=selection).constraint_selection is selection


def test_stage_reads_solve_options_from_context():
    """AllocationStage reads constraint_selection, time_limit_s and solver_log from context."""
    cs = ConstraintSelection(dma_channels=False)
    ctx = StageContext.from_kwargs(
        steady_state_problem=MagicMock(),
        output_path="/tmp/test",
        backend="ORTOOLS_GSCIP",
        constraint_selection=cs,
        time_limit_s=12.5,
        solver_log=True,
    )
    stage = AllocationStage([MagicMock()], ctx)
    assert stage.constraint_selection.dma_channels is False, "Stage must read constraint_selection from context"
    assert stage.time_limit_s == 12.5
    assert stage.solver_log is True


def test_stage_defaults_when_absent():
    """AllocationStage defaults to ConstraintSelection(), a 300 s limit and a silent solver."""
    ctx = StageContext.from_kwargs(
        steady_state_problem=MagicMock(),
        output_path="/tmp/test",
        backend="ORTOOLS_GSCIP",
    )
    stage = AllocationStage([MagicMock()], ctx)
    assert stage.constraint_selection == ConstraintSelection(), (
        "Stage must default constraint_selection to ConstraintSelection() (all True)"
    )
    assert stage.time_limit_s == 300
    assert stage.solver_log is False
