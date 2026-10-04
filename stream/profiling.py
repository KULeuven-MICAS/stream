"""Wall-clock spans of a run, recorded only while a :class:`Profile` is active.

A span costs one context-variable read when no profile is active, so stages and the allocation model can
mark their phases unconditionally. Spans nest: each records its inclusive time and its own
(exclusive) time, keyed by the path of the spans enclosing it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stream.stages.context import StageContext
    from stream.stages.stage import StageCallable


@dataclass
class SpanStats:
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0


@dataclass
class Profile:
    """Span statistics by path, outermost name first."""

    spans: dict[tuple[str, ...], SpanStats] = field(default_factory=dict)
    _stack: list[list[Any]] = field(default_factory=list, repr=False)

    def by_name(self) -> dict[str, SpanStats]:
        """The same statistics summed over every path that ends in each name."""
        totals: dict[str, SpanStats] = {}
        for path, stats in self.spans.items():
            total = totals.setdefault(path[-1], SpanStats())
            total.calls += stats.calls
            total.exclusive_ns += stats.exclusive_ns
            if path[-1] not in path[:-1]:
                total.inclusive_ns += stats.inclusive_ns
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [
                {
                    "path": list(path),
                    "calls": s.calls,
                    "inclusive_s": s.inclusive_ns / 1e9,
                    "exclusive_s": s.exclusive_ns / 1e9,
                }
                for path, s in self.spans.items()
            ]
        }


_ACTIVE: ContextVar[Profile | None] = ContextVar("stream_profile", default=None)


@contextmanager
def profile() -> Iterator[Profile]:
    """Record every span opened in this block; profiles do not nest."""
    if _ACTIVE.get() is not None:
        raise RuntimeError("a profile is already active")
    active = Profile()
    token = _ACTIVE.set(active)
    try:
        yield active
    finally:
        _ACTIVE.reset(token)


@contextmanager
def span(name: str) -> Iterator[None]:
    """Time the enclosed block as ``name`` within whatever span encloses it."""
    active = _ACTIVE.get()
    if active is None:
        yield
        return
    stack = active._stack
    path = (*stack[-1][0], name) if stack else (name,)
    frame = [path, 0]
    stack.append(frame)
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns() - start
        stack.pop()
        if stack:
            stack[-1][1] += elapsed
        stats = active.spans.setdefault(path, SpanStats())
        stats.calls += 1
        stats.inclusive_ns += elapsed
        stats.exclusive_ns += elapsed - frame[1]


class _TimedStage:
    """A stage whose every resume is a span named after it, so a stage's own time excludes its sub-stages."""

    def __init__(self, stage_callable: StageCallable, list_of_callables: list[StageCallable], ctx: StageContext):
        self.name = getattr(stage_callable, "__name__", type(stage_callable).__name__)
        with span(self.name):
            self.stage = stage_callable(list_of_callables, ctx)

    def is_leaf(self) -> bool:
        return self.stage.is_leaf()

    def run(self) -> Iterator[StageContext]:
        results = self.stage.run()
        while True:
            with span(self.name):
                try:
                    ctx = next(results)
                except StopIteration:
                    return
            yield ctx


def timed(stage_callable: StageCallable) -> StageCallable:
    def build(list_of_callables: list[StageCallable], ctx: StageContext) -> _TimedStage:
        return _TimedStage(stage_callable, list_of_callables, ctx)

    build.__name__ = getattr(stage_callable, "__name__", "stage")
    return build  # type: ignore[return-value]


class TimingInstrumentation:
    """The ``timing`` observer: profiles a run's stages and the spans inside them, and writes the profile
    as JSON to ``path`` when the run ends."""

    def __init__(self, *, run_name: str, path: str | None = None) -> None:
        self.run_name = run_name
        self.path = path
        self._context = profile()
        self.profile = self._context.__enter__()

    def instrument(self, stages: list[StageCallable]) -> list[StageCallable]:
        return [timed(stage) for stage in stages]

    def finish(self) -> None:
        self._close(None)

    def fail(self, reason: str) -> None:
        self._close(reason)

    def _close(self, failure: str | None) -> None:
        self._context.__exit__(None, None, None)
        if self.path:
            report = {"run": self.run_name, "failure": failure, **self.profile.to_dict()}
            Path(self.path).write_text(json.dumps(report, indent=1) + "\n")
