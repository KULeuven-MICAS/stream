"""Unit tests for pipeline threading of the solve options into the allocation stage.

Covers:
  - SolveOptions carries the constraint families, None (the default set) by default
  - AllocationStage reads the families, the time limit and the solver log from context
  - AllocationStage defaults to the problem's default families, 300 s and no solver log when absent
"""

from unittest.mock import MagicMock

from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES, drop_families, load_families
from stream.stages.allocation.steady_state_allocation import AllocationStage
from stream.stages.context import StageContext


def test_solve_options_carry_families():
    from stream.api import SolveOptions

    assert SolveOptions().families is None
    assert SolveOptions(families=["placement"]).families == ["placement"]


def test_stage_reads_solve_options_from_context():
    """AllocationStage reads families, time_limit_s and solver_log from context."""
    families = load_families(drop_families(DEFAULT_FAMILIES, ["dma_channels"]))
    ctx = StageContext.from_kwargs(
        steady_state_problem=MagicMock(),
        output_path="/tmp/test",
        backend="ORTOOLS_GSCIP",
        families=families,
        time_limit_s=12.5,
        solver_log=True,
    )
    stage = AllocationStage([MagicMock()], ctx)
    assert stage.families is families, "Stage must read the families from context"
    assert stage.time_limit_s == 12.5
    assert stage.solver_log is True


def test_stage_defaults_when_absent():
    """AllocationStage defaults to the problem's default families, a 300 s limit and a silent solver."""
    problem = MagicMock()
    problem.transfer_context.default_families = DEFAULT_FAMILIES
    ctx = StageContext.from_kwargs(steady_state_problem=problem, output_path="/tmp/test", backend="ORTOOLS_GSCIP")
    stage = AllocationStage([MagicMock()], ctx)
    assert [name for name, _ in stage.families.specs()] == list(DEFAULT_FAMILIES)
    assert stage.time_limit_s == 300
    assert stage.solver_log is False
