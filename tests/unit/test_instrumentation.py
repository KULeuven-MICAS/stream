"""The observation hook: enabled by name, per call, and never load-bearing.

A pipeline must run identically whether or not an observer is installed, and an observer that is
absent, broken, or throwing must not change the outcome of a solve.
"""

from __future__ import annotations

from stream import instrumentation as instr
from stream.instrumentation import build_instrumentation, fail_instrumentation, finish_instrumentation, instrument
from stream.plugins import LoadedPlugin


class _Recorder:
    """A minimal observer: records what it was told and appends a marker to the stage list."""

    def __init__(self, *, run_name: str, marker: str = "probe"):
        self.run_name = run_name
        self.marker = marker
        self.finished = False
        self.failure: str | None = None

    def instrument(self, stages):
        return [*stages, self.marker]

    def finish(self):
        self.finished = True

    def fail(self, reason):
        self.failure = reason


def _register(monkeypatch, **factories):
    plugins = [LoadedPlugin(name, obj, "stream-overlay", 10) for name, obj in factories.items()]
    monkeypatch.setattr(instr, "load_group", lambda group: plugins)


def test_nothing_is_instrumented_unless_asked(monkeypatch):
    _register(monkeypatch, progress=_Recorder)
    assert build_instrumentation("run", None) == []
    assert build_instrumentation("run", {}) == []


def test_an_observer_is_built_by_name_and_given_its_options(monkeypatch):
    _register(monkeypatch, progress=_Recorder)
    observers = build_instrumentation("my-run", {"progress": {"marker": "P"}})
    assert len(observers) == 1
    assert observers[0].run_name == "my-run"
    assert observers[0].marker == "P"


def test_an_unknown_name_is_a_no_op(monkeypatch):
    """A pipeline must not fail because an optional observer is absent from the environment."""
    _register(monkeypatch, progress=_Recorder)
    assert build_instrumentation("run", {"not-installed": {}}) == []


def test_a_factory_that_raises_is_skipped(monkeypatch):
    def exploding(**_kwargs):
        raise RuntimeError("bad observer")

    _register(monkeypatch, progress=_Recorder, broken=exploding)
    observers = build_instrumentation("run", {"broken": {}, "progress": {}})
    assert [type(o).__name__ for o in observers] == ["_Recorder"]


def test_instrument_applies_every_observer_in_order(monkeypatch):
    _register(monkeypatch, a=_Recorder, b=_Recorder)
    observers = build_instrumentation("run", {"a": {"marker": "A"}, "b": {"marker": "B"}})
    assert instrument(["stage"], observers) == ["stage", "A", "B"]


def test_a_throwing_observer_leaves_the_stage_list_usable():
    class Hostile:
        def instrument(self, stages):
            raise RuntimeError("nope")

        def finish(self): ...
        def fail(self, reason): ...

    assert instrument(["stage"], [Hostile()]) == ["stage"]


def test_finish_and_fail_reach_every_observer(monkeypatch):
    _register(monkeypatch, a=_Recorder, b=_Recorder)
    observers = build_instrumentation("run", {"a": {}, "b": {}})

    finish_instrumentation(observers)
    assert all(o.finished for o in observers)

    fail_instrumentation(observers, "solver died")
    assert all(o.failure == "solver died" for o in observers)


def test_a_throwing_observer_does_not_break_finish_or_fail():
    class Hostile:
        def instrument(self, stages):
            return stages

        def finish(self):
            raise RuntimeError("nope")

        def fail(self, reason):
            raise RuntimeError("nope")

    finish_instrumentation([Hostile()])
    fail_instrumentation([Hostile()], "reason")
