"""The ``accelforge`` backend with AccelForge: nodes mapped on cores, a run that names it pricing what ZigZag prices,
its cache, and AccelForge leaving the process's logging, warnings and environment as they were."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stream.hardware.ports import OUTPUT, READ_BY_DATAPATH, WRITE_BY_DATAPATH, input_role
from stream.stages.estimation.accelforge.backend import (
    ACCELFORGE_BACKEND,
    AccelForgeEstimationError,
    AccelForgeEstimator,
    AccelForgeOptions,
)
from stream.stages.estimation.accelforge.workload import describe
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator

from ..cost_stage_runs import inputs, priced
from .cores import conv, example_core, matmul

mapper = pytest.importorskip("stream.stages.estimation.accelforge.mapper", reason="accelforge is not installed")
accelforge = pytest.importorskip("accelforge")

REPO = Path(__file__).parents[3]


@pytest.mark.parametrize("window", [3, 1])
def test_accelforge_maps_every_iteration_of_a_strided_conv(window):
    spec = accelforge.Spec.from_yaml(accelforge.examples.arches.simple)
    spec.workload = mapper.to_workload(describe(conv(2, window), (8, 4, 8, 8, 3, window)))
    spec.mapper.metrics = accelforge.Metrics.ENERGY
    spec.mapper.explore_loop_orders = False
    macs = spec.map_workload_to_arch(print_progress=False).actions(per_component=True)[("MAC", "compute")]
    assert macs == 8 * 4 * 8 * 8 * 3 * window


def test_a_free_order_template_accelforge_rejects_is_mapped_again_in_forced_order():
    mapped = mapper.map_node(describe(matmul(), (64, 64, 64)), example_core("meta_prototype"), "free", False)
    assert mapped.memory_order == "forced"
    assert mapped.fallback.startswith("free order refused, mapped in forced order: AssertionError: BUG")


def test_a_core_it_cannot_express_fails_naming_node_and_core():
    workload, accelerator, mapping = inputs()
    node, memory = workload.get_computation_nodes()[0], example_core("tpu_v7_vmem")
    estimator = AccelForgeEstimator(ZigZagCostEstimator(workload, accelerator, mapping), AccelForgeOptions())
    with pytest.raises(AccelForgeEstimationError, match=rf"could not cost node {node.name} on core {memory.id}"):
        estimator.estimate(node, memory)


def _entries(stage: CoreCostEstimationStage) -> dict:
    return {(node.name, core.id): entry for node, per in stage.cost_lut.lut.items() for core, entry in per.items()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[Path, CoreCostEstimationStage, CoreCostEstimationStage]:
    """The 2-conv workload on four TPU-like cores costed by ZigZag, and by AccelForge in a cache of its own."""
    zigzag = priced(tmp_path_factory.mktemp("zigzag"))
    output = tmp_path_factory.mktemp("accelforge")
    return output, zigzag, priced(output, core_cost_backend="accelforge")


@pytest.mark.slow
def test_a_named_run_prices_what_zigzag_prices_with_the_traffic_and_layouts_the_allocation_reads(runs):
    _, zigzag, accelforge_run = runs
    zigzag_entries, entries = _entries(zigzag), _entries(accelforge_run)
    assert zigzag_entries.keys() == entries.keys() and len(entries) == 8
    assert {e.metadata["backend"] for e in zigzag_entries.values()} == {"zigzag"}
    for key, entry in entries.items():
        assert entry.metadata["backend"] == "accelforge"
        assert entry.ideal_cycle == zigzag_entries[key].ideal_cycle
        assert entry.latency_total >= entry.ideal_temporal_cycle >= entry.ideal_cycle
        moved = {(role, way) for role, way, bits in ACCELFORGE_BACKEND.port_traffic(entry) if bits > 0}
        assert {
            (input_role(1), READ_BY_DATAPATH),
            (input_role(2), READ_BY_DATAPATH),
            (OUTPUT, WRITE_BY_DATAPATH),
        } <= moved
        assert len(ACCELFORGE_BACKEND.contiguous_axes(entry)) == len(entry.layer.operand_mapping)


@pytest.mark.slow
def test_its_cached_entries_are_reused_with_the_same_options_and_repriced_with_others(runs, monkeypatch):
    output, _, _ = runs
    calls: list[object] = []
    real = mapper.map_node
    monkeypatch.setattr(mapper, "map_node", lambda *args: calls.append(args) or real(*args))
    priced(output, core_cost_backend="accelforge", core_cost_options={"accelforge": {"memory_order": "free"}})
    assert calls == [], "the default options again, so the cache serves the run"
    stage = priced(output, core_cost_backend="accelforge", core_cost_options={"accelforge": {"memory_order": "forced"}})
    assert calls, "another memory order decides other entries"
    assert {e.metadata["memory_order"] for e in _entries(stage).values()} == {"forced"}


IMPORT = """
import logging, os, sys, warnings
state = lambda: (list(logging.root.handlers), list(warnings.filters), dict(os.environ))
before = state()
import stream.stages.estimation.accelforge.mapper
print("accelforge" in sys.modules, state() == before)
"""


def test_importing_accelforge_leaves_logging_warnings_and_the_environment_as_they_were():
    result = subprocess.run([sys.executable, "-c", IMPORT], capture_output=True, text=True, cwd=REPO, check=True)
    assert result.stdout.split() == ["True", "True"]
