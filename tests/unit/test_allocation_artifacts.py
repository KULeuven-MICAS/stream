"""The files of an allocation solve are written only when asked for."""

import tomllib
from importlib import import_module
from pathlib import Path

from stream.allocation.artifacts import AllocationArtifacts, artifacts
from stream.api import evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
ARTIFACTS = {
    "optimization_metrics.yaml",
    "slot_latency_breakdown.yaml",
    "steady_state_trace.json",
    "steady_state_trace_compact.json",
    "steady_state_workload_final.svg",
}


def _solve(path: Path, two_conv: TwoConvWorkloadConfig) -> set[str]:
    evaluate_mapping(ACCELERATOR, make_2_conv_workload(two_conv), str(path), make_2_conv_mapping(two_conv))
    return {file.name for file in path.rglob("tetra/*")}


def test_a_solve_writes_no_artifacts_by_default(tmp_path: Path, two_conv: TwoConvWorkloadConfig) -> None:
    assert not _solve(tmp_path, two_conv) & ARTIFACTS


def test_a_solve_writes_its_artifacts_when_asked(tmp_path: Path, two_conv: TwoConvWorkloadConfig) -> None:
    with artifacts():
        written = _solve(tmp_path, two_conv)
    assert ARTIFACTS <= written


def test_the_observer_is_registered_as_allocation_artifacts() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    target = pyproject["project"]["entry-points"]["stream.instrumentation"]["allocation_artifacts"]
    module, name = target.split(":")
    assert getattr(import_module(module), name) is AllocationArtifacts
