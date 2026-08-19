"""Observation of a running pipeline, supplied out of tree via the ``stream.instrumentation`` group."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from stream.plugins import load_group

if TYPE_CHECKING:
    from stream.stages.stage import StageCallable

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "stream.instrumentation"


@runtime_checkable
class StageInstrumentation(Protocol):
    """Wraps a stage list to observe a run, and is told how the run ended."""

    def instrument(self, stages: list[StageCallable]) -> list[StageCallable]:
        """Return the stage list to run -- typically the original with observers interleaved."""
        ...

    def finish(self) -> None:
        """The run completed."""
        ...

    def fail(self, reason: str) -> None:
        """The run stopped early. Called before the exception is re-raised, never in place of it."""
        ...


def build_instrumentation(run_name: str, spec: dict[str, Any] | None) -> list[StageInstrumentation]:
    """Instantiate the observers named in ``spec`` ({name: options}); empty when nothing is requested."""
    if not spec:
        return []
    factories = {plugin.name: plugin.obj for plugin in load_group(ENTRY_POINT_GROUP)}
    built: list[StageInstrumentation] = []
    for name, options in spec.items():
        factory = factories.get(name)
        if factory is None:
            logger.warning("no instrumentation registered as %r; continuing without it", name)
            continue
        try:
            observer = factory(run_name=run_name, **(options or {}))
        except Exception as exc:  # noqa: BLE001 -- an observer must not be able to fail a solve
            logger.warning("skipping %r instrumentation: %s", name, exc)
            continue
        if observer is not None:
            built.append(observer)
    return built


def instrument(stages: list[StageCallable], observers: list[StageInstrumentation]) -> list[StageCallable]:
    """Apply every observer to the stage list, in order."""
    for observer in observers:
        try:
            stages = observer.instrument(stages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("instrumentation %r could not wrap the stage list: %s", type(observer).__name__, exc)
    return stages


def finish_instrumentation(observers: list[StageInstrumentation]) -> None:
    for observer in observers:
        try:
            observer.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("instrumentation %r failed on finish: %s", type(observer).__name__, exc)


def fail_instrumentation(observers: list[StageInstrumentation], reason: str) -> None:
    for observer in observers:
        try:
            observer.fail(reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("instrumentation %r failed on fail: %s", type(observer).__name__, exc)
