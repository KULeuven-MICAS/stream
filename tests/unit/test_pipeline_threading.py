"""Unit tests for pipeline threading of the constraint_selection parameter.

Covers:
  - SolveOptions carries constraint_selection, None by default
  - ConstraintOptimizationAllocationStage reads constraint_selection from context
  - ConstraintOptimizationAllocationStage defaults to ConstraintSelection() when absent
  - SteadyStateScheduler stores constraint_selection from constructor kwarg
  - SteadyStateScheduler defaults constraint_selection to None when omitted
"""

from unittest.mock import MagicMock

from stream.opt.solver import ConstraintSelection


def test_solve_options_carry_constraint_selection():
    from stream.api import SolveOptions

    assert SolveOptions().constraint_selection is None
    selection = ConstraintSelection(dma_channels=False)
    assert SolveOptions(constraint_selection=selection).constraint_selection is selection


# ---------------------------------------------------------------------------
# Stage reads constraint_selection from context when present
# ---------------------------------------------------------------------------


def test_stage_reads_constraint_selection_from_context():
    """ConstraintOptimizationAllocationStage reads constraint_selection from context."""
    from stream.stages.allocation.constraint_optimization_allocation import (
        ConstraintOptimizationAllocationStage,
    )
    from stream.stages.context import StageContext

    cs = ConstraintSelection(dma_channels=False)
    ctx = StageContext.from_kwargs(
        workload=MagicMock(),
        accelerator=MagicMock(),
        mapping=MagicMock(),
        cost_lut=MagicMock(),
        fusion_splits={},
        output_path="/tmp/test",
        backend="ORTOOLS_GSCIP",
        constraint_selection=cs,
    )
    stage = ConstraintOptimizationAllocationStage([MagicMock()], ctx)
    assert stage.constraint_selection.dma_channels is False, "Stage must read constraint_selection from context"


# ---------------------------------------------------------------------------
# Stage defaults to ConstraintSelection() when absent from context
# ---------------------------------------------------------------------------


def test_stage_defaults_constraint_selection_when_absent():
    """ConstraintOptimizationAllocationStage defaults to ConstraintSelection() when key is absent."""
    from stream.stages.allocation.constraint_optimization_allocation import (
        ConstraintOptimizationAllocationStage,
    )
    from stream.stages.context import StageContext

    ctx = StageContext.from_kwargs(
        workload=MagicMock(),
        accelerator=MagicMock(),
        mapping=MagicMock(),
        cost_lut=MagicMock(),
        fusion_splits={},
        output_path="/tmp/test",
        backend="ORTOOLS_GSCIP",
        # No constraint_selection key
    )
    stage = ConstraintOptimizationAllocationStage([MagicMock()], ctx)
    assert stage.constraint_selection == ConstraintSelection(), (
        "Stage must default constraint_selection to ConstraintSelection() (all True)"
    )


# ---------------------------------------------------------------------------
# SteadyStateScheduler stores constraint_selection when provided
# ---------------------------------------------------------------------------


def test_scheduler_stores_constraint_selection():
    """SteadyStateScheduler stores constraint_selection when passed as kwarg."""
    from stream.cost_model.steady_state_scheduler import SteadyStateScheduler

    cs = ConstraintSelection(buffer_descriptors=False)
    scheduler = SteadyStateScheduler(
        MagicMock(),  # workload
        MagicMock(),  # accelerator
        MagicMock(),  # mapping
        {},  # fusion_splits
        MagicMock(),  # cost_lut
        constraint_selection=cs,
    )
    assert scheduler.constraint_selection is cs, "SteadyStateScheduler must store constraint_selection"
    assert scheduler.constraint_selection.buffer_descriptors is False


# ---------------------------------------------------------------------------
# SteadyStateScheduler defaults constraint_selection to None when omitted
# ---------------------------------------------------------------------------


def test_scheduler_defaults_constraint_selection_when_none():
    """SteadyStateScheduler defaults constraint_selection to None when not passed."""
    from stream.cost_model.steady_state_scheduler import SteadyStateScheduler

    scheduler = SteadyStateScheduler(
        MagicMock(),  # workload
        MagicMock(),  # accelerator
        MagicMock(),  # mapping
        {},  # fusion_splits
        MagicMock(),  # cost_lut
    )
    assert scheduler.constraint_selection is None, "SteadyStateScheduler constraint_selection must default to None"
