"""An allocation solve writes its reports, traces and figures beside its group unless a sweep turns them off."""

from pathlib import Path

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
ARTIFACTS = {
    "reports/optimization_metrics.yaml",
    "reports/slot_latency_breakdown.yaml",
    "traces/steady_state_trace.json",
    "traces/steady_state_trace_compact.json",
    "figures/steady_state_workload_final.svg",
}


def _solve(path: Path, two_conv: TwoConvWorkloadConfig, options: SolveOptions) -> set[str]:
    workload, mapping = make_2_conv_workload(two_conv), make_2_conv_mapping(two_conv)
    evaluate_mapping(ACCELERATOR, workload, str(path), mapping, options)
    return {str(file.relative_to(path / "group_0" / "allocation")) for file in path.glob("group_0/allocation/*/*")}


def test_a_solve_writes_its_artifacts_by_default(tmp_path: Path, two_conv: TwoConvWorkloadConfig) -> None:
    assert ARTIFACTS <= _solve(tmp_path, two_conv, SolveOptions())


def test_a_sweep_can_turn_the_artifacts_off(tmp_path: Path, two_conv: TwoConvWorkloadConfig) -> None:
    assert not _solve(tmp_path, two_conv, SolveOptions(artifacts=False))
