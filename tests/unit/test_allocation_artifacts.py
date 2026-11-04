"""An allocation solve writes its reports, traces and figures beside its group unless a sweep turns them off."""

from pathlib import Path

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.stages.context import StageContext
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
TWO_CONV_ONNX = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
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
    assert {"hardware_graph.svg", "group_0/tiled_workload.svg"} <= {
        str(file.relative_to(tmp_path)) for file in tmp_path.rglob("*.svg")
    }


def test_a_sweep_can_turn_the_artifacts_off(tmp_path: Path, two_conv: TwoConvWorkloadConfig) -> None:
    """No artifacts are written, and no picture."""
    assert not _solve(tmp_path, two_conv, SolveOptions(artifacts=False))
    assert not list(tmp_path.rglob("*.svg"))


@pytest.mark.parametrize("artifacts", [True, False])
def test_the_onnx_parser_draws_the_workload_only_with_the_artifacts(tmp_path: Path, artifacts: bool) -> None:
    ctx = StageContext.from_kwargs(workload_path=TWO_CONV_ONNX, output_path=str(tmp_path), artifacts=artifacts)
    MainStage([ONNXModelParserStage, LeafStage], ctx).run()
    assert (tmp_path / "workload_graph.svg").exists() == artifacts


def test_a_tile_search_leaves_the_chosen_candidates_artifacts_beside_its_group(
    tmp_path: Path, two_conv: TwoConvWorkloadConfig
) -> None:
    assert ARTIFACTS <= _solve(tmp_path, two_conv, SolveOptions(tile_search=True))
    assert (tmp_path / "group_0" / "core_cost_lut.yaml").exists()
