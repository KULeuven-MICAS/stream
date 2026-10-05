"""Spans: free when nothing profiles, nested and self-timed when something does."""

import time

import pytest

from stream.profiling import profile, span
from stream.stages.stage import LeafStage, Stage


def test_a_span_records_nothing_outside_a_profile():
    with span("idle"):
        pass
    with profile() as recorded:
        pass
    assert recorded.spans == {}


def test_a_span_counts_its_own_time_apart_from_the_spans_inside_it():
    with profile() as recorded:
        with span("outer"):
            time.sleep(0.01)
            for _ in range(2):
                with span("inner"):
                    time.sleep(0.02)
    outer, inner = recorded.spans[("outer",)], recorded.spans[("outer", "inner")]
    assert inner.calls == 2
    assert outer.inclusive_ns >= outer.exclusive_ns + inner.inclusive_ns
    assert 0 < outer.exclusive_ns < inner.inclusive_ns


def test_profiles_do_not_nest():
    with profile(), pytest.raises(RuntimeError, match="already active"):
        with profile():
            pass


class _Outer(Stage):
    def run(self):
        time.sleep(0.01)
        yield from self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run()


class _Leaf(LeafStage):
    def run(self):
        time.sleep(0.02)
        yield self.ctx


def test_a_stage_is_a_span_that_excludes_the_stages_it_runs():
    with profile() as recorded:
        (result,) = list(_Outer([_Leaf], "ctx").run())
    assert result == "ctx"
    outer, leaf = recorded.spans[("_Outer",)], recorded.spans[("_Outer", "_Leaf")]
    assert outer.exclusive_ns < leaf.exclusive_ns
    assert outer.inclusive_ns >= outer.exclusive_ns + leaf.inclusive_ns


def test_a_solve_that_fails_before_its_stages_run_closes_its_profile(monkeypatch, tmp_path):
    from stream import api
    from stream.profiling import TimingInstrumentation

    monkeypatch.setattr(api, "build_instrumentation", lambda *_: [TimingInstrumentation(run_name="run")])
    with pytest.raises(FileNotFoundError):
        api.evaluate_mapping(str(tmp_path / "missing.yaml"), "workload.onnx", str(tmp_path))
    with profile():
        pass
