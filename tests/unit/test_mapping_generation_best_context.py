"""The mapping search must yield the winning variant's context, not the last one's.

Every variant runs the inner pipeline against the ONE shared StageContext, so keeping a reference
to it after the loop describes whichever variant finished last. Only the float latency was right.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from stream.stages.context import StageContext
from stream.stages.generation import mapping_generation
from stream.stages.generation.mapping_generation import MappingGenerationStage

LATENCIES = [300.0, 100.0, 700.0]  # best is variant 1; the LAST evaluated is variant 2
BEST_INDEX = 1


class _FakeGenerator:
    """Stands in for MappingGenerator: emits one trivial variant per entry in LATENCIES."""

    def __init__(self, n: int):
        self.n = n

    def run(self):
        for i in range(self.n):
            yield i, [("Gemm_Left", [])], {"variant": i}

    def save_mapping(self, *, mapping, variant, idx, output_dir):
        path = os.path.join(output_dir, f"{idx}_mapping.yaml")
        with open(path, "w") as f:
            f.write("fused_groups: []\n")
        return path


class _FakeSubStage:
    """Stands in for the inner pipeline: mutates the shared context in place, exactly as the real
    stages do (ctx.set(scheduler=...) on the context handed down to them)."""

    def __init__(self, list_of_callables, ctx):
        self.ctx = ctx

    def run(self):
        idx = int(os.path.basename(self.ctx.get("output_path")))
        self.ctx.set(scheduler=SimpleNamespace(latency_total=LATENCIES[idx]), variant=idx)
        yield self.ctx


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping_generation, "MappingGenerator", lambda **kwargs: _FakeGenerator(len(LATENCIES)))
    ctx = StageContext.from_kwargs(accelerator=object(), workload=object(), output_path=str(tmp_path))
    return MappingGenerationStage([_FakeSubStage], ctx), ctx


def test_yielded_context_describes_the_winning_variant(stage):
    mapping_stage, shared_ctx = stage

    (best,) = list(mapping_stage.run())

    assert best.get("best_latency") == min(LATENCIES)
    assert best.get("scheduler").latency_total == best.get("best_latency")
    assert best.get("variant") == BEST_INDEX
    # The shared context has moved on to the last variant; the snapshot must not follow it.
    assert shared_ctx.get("scheduler").latency_total == LATENCIES[-1]


def test_winning_variant_is_identifiable(stage):
    mapping_stage, _ = stage

    (best,) = list(mapping_stage.run())

    assert best.get("best_mapping_index") == BEST_INDEX
    assert os.path.basename(os.path.dirname(best.get("best_mapping_path"))) == str(BEST_INDEX)
