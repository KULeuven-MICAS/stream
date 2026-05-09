"""Unit tests for pipeline threading of constraint_selection parameter.

Covers Phase 6 Plan 01 requirements (PIPE-01, UI-01):
  - optimize_allocation_co and optimize_mapping accept constraint_selection kwarg
  - ConstraintOptimizationAllocationStage reads constraint_selection from context
  - ConstraintOptimizationAllocationStage defaults to ConstraintSelection() when absent
  - SteadyStateScheduler stores constraint_selection from constructor kwarg
  - SteadyStateScheduler defaults constraint_selection to None when omitted
  - CLI conversion pattern: list of disabled names -> ConstraintSelection instance

Covers Phase 6 Plan 02 requirements (UI-02):
  - --disable-constraints CLI flag parsing
  - Conversion of disabled list to ConstraintSelection instance
"""

import argparse
import inspect
from unittest.mock import MagicMock

from stream.opt.solver import ConstraintSelection

# ---------------------------------------------------------------------------
# Test 1: optimize_allocation_co accepts constraint_selection kwarg
# ---------------------------------------------------------------------------


def test_api_optimize_allocation_co_accepts_constraint_selection():
    """optimize_allocation_co signature has constraint_selection param with default None."""
    from stream.api import optimize_allocation_co

    sig = inspect.signature(optimize_allocation_co)
    assert "constraint_selection" in sig.parameters, "optimize_allocation_co must accept 'constraint_selection' kwarg"
    assert sig.parameters["constraint_selection"].default is None, "constraint_selection must default to None"


# ---------------------------------------------------------------------------
# Test 2: optimize_mapping accepts constraint_selection kwarg
# ---------------------------------------------------------------------------


def test_api_optimize_mapping_accepts_constraint_selection():
    """optimize_mapping signature has constraint_selection param with default None."""
    from stream.api import optimize_mapping

    sig = inspect.signature(optimize_mapping)
    assert "constraint_selection" in sig.parameters, "optimize_mapping must accept 'constraint_selection' kwarg"
    assert sig.parameters["constraint_selection"].default is None, "constraint_selection must default to None"


# ---------------------------------------------------------------------------
# Test 3: Stage reads constraint_selection from context when present
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
# Test 4: Stage defaults to ConstraintSelection() when absent from context
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
# Test 5: SteadyStateScheduler stores constraint_selection when provided
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
# Test 6: SteadyStateScheduler defaults constraint_selection to None when omitted
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


# ---------------------------------------------------------------------------
# Test 7: CLI disable-constraints conversion pattern
# ---------------------------------------------------------------------------


def test_cli_disable_constraints_parsing():
    """CLI conversion: list of disabled names -> ConstraintSelection with those fields False."""
    # Simulate what CLI scripts will do with --disable-constraints flag
    disabled_list = ["memory_capacity", "dma_channels"]
    disabled_set = set(disabled_list)

    cs = ConstraintSelection(
        memory_capacity="memory_capacity" not in disabled_set,
        object_fifo_depth="object_fifo_depth" not in disabled_set,
        buffer_descriptors="buffer_descriptors" not in disabled_set,
        dma_channels="dma_channels" not in disabled_set,
    )

    assert cs.memory_capacity is False, "memory_capacity should be False (was in disabled list)"
    assert cs.dma_channels is False, "dma_channels should be False (was in disabled list)"
    assert cs.object_fifo_depth is True, "object_fifo_depth should be True (not in disabled list)"
    assert cs.buffer_descriptors is True, "buffer_descriptors should be True (not in disabled list)"


# ---------------------------------------------------------------------------
# CLI --disable-constraints parsing (UI-02)
# ---------------------------------------------------------------------------


def _make_disable_constraints_parser():
    """Build a minimal parser with --disable-constraints matching the main scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-constraints",
        nargs="*",
        choices=["memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels"],
        default=[],
        metavar="CONSTRAINT",
        help="Disable hardware resource constraint groups.",
    )
    return parser


def _build_constraint_selection_from_disabled(disabled_list):
    """Convert a list of disabled constraint names to a ConstraintSelection."""
    disabled = set(disabled_list or [])
    return ConstraintSelection(
        memory_capacity="memory_capacity" not in disabled,
        object_fifo_depth="object_fifo_depth" not in disabled,
        buffer_descriptors="buffer_descriptors" not in disabled,
        dma_channels="dma_channels" not in disabled,
    )


def test_cli_disable_constraints_two_fields():
    """Parsing --disable-constraints with two fields produces correct ConstraintSelection."""
    parser = _make_disable_constraints_parser()
    args = parser.parse_args(["--disable-constraints", "memory_capacity", "dma_channels"])
    assert args.disable_constraints == ["memory_capacity", "dma_channels"]
    cs = _build_constraint_selection_from_disabled(args.disable_constraints)
    assert cs.memory_capacity is False
    assert cs.object_fifo_depth is True
    assert cs.buffer_descriptors is True
    assert cs.dma_channels is False


def test_cli_disable_constraints_absent():
    """When --disable-constraints is not passed, result is all-True ConstraintSelection."""
    parser = _make_disable_constraints_parser()
    args = parser.parse_args([])
    assert args.disable_constraints == []
    cs = _build_constraint_selection_from_disabled(args.disable_constraints)
    assert cs == ConstraintSelection()  # all True


def test_cli_disable_constraints_flag_no_values():
    """When --disable-constraints is passed with no values (nargs='*'), result is all-True."""
    parser = _make_disable_constraints_parser()
    args = parser.parse_args(["--disable-constraints"])
    assert args.disable_constraints == []
    cs = _build_constraint_selection_from_disabled(args.disable_constraints)
    assert cs == ConstraintSelection()


def test_cli_disable_constraints_all_four():
    """Passing all four constraint names disables all four fields."""
    parser = _make_disable_constraints_parser()
    args = parser.parse_args(
        ["--disable-constraints", "memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels"]
    )
    cs = _build_constraint_selection_from_disabled(args.disable_constraints)
    assert cs.memory_capacity is False
    assert cs.object_fifo_depth is False
    assert cs.buffer_descriptors is False
    assert cs.dma_channels is False


def test_cli_disable_constraints_single_field():
    """Passing a single constraint name disables only that field."""
    parser = _make_disable_constraints_parser()
    args = parser.parse_args(["--disable-constraints", "object_fifo_depth"])
    cs = _build_constraint_selection_from_disabled(args.disable_constraints)
    assert cs.memory_capacity is True
    assert cs.object_fifo_depth is False
    assert cs.buffer_descriptors is True
    assert cs.dma_channels is True
