"""The files a solve can write beside its result -- the Perfetto traces of the schedule, the solver's progress
and metrics, the slot latency breakdown and a picture of the solved workload. They are written only while an
``allocation_artifacts`` observer, or :func:`artifacts`, is active, so a sweep pays nothing for them."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from stream.profiling import span
from stream.visualization.steady_state_trace import export_steady_state_trace

if TYPE_CHECKING:
    from stream.allocation.schedule import SteadyStateSchedule
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.stages.stage import StageCallable

logger = logging.getLogger(__name__)

_ACTIVE: ContextVar[bool] = ContextVar("stream_allocation_artifacts", default=False)


@contextmanager
def artifacts() -> Iterator[None]:
    """Write the artifacts of every allocation solved in this block."""
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def write_artifacts(allocator: TransferAndTensorAllocator, schedule: SteadyStateSchedule) -> None:
    """Write the artifacts of a solved allocation into the allocator's output path, if they are asked for."""
    if not _ACTIVE.get():
        return
    output_path = allocator.output_path
    os.makedirs(output_path, exist_ok=True)
    with span("milp_files"):
        allocator.plot_optimization_progress(
            show=False, save_path=os.path.join(output_path, "optimization_progress.png")
        )
        allocator.save_optimization_trace(os.path.join(output_path, "optimization_trace.yaml"))
        allocator.save_optimization_metrics(save_path=os.path.join(output_path, "optimization_metrics.yaml"))
        allocator.save_slot_latency_breakdown(save_path=os.path.join(output_path, "slot_latency_breakdown.yaml"))
    with span("trace_export"):
        latency = schedule.solution.latency
        fname = ""
        trace_path = ""
        try:
            for compact, fname in [(True, "steady_state_trace_compact.json"), (False, "steady_state_trace.json")]:
                trace_path = export_steady_state_trace(
                    allocator.context,
                    schedule.solution.transfer_routes,
                    iterations=schedule.iterations,
                    overlap=latency.overlap,
                    latency_per_iteration=latency.per_iteration,
                    output_path=output_path,
                    compact=compact,
                    filename=fname,
                )
            logger.info("Steady-state schedule trace: %s", trace_path)
        except Exception as exc:  # never let a visualisation failure abort the run
            logger.warning("Failed to export steady-state trace (%s): %s", fname, exc)
    with span("visualize"):
        schedule.workload.visualize(
            os.path.join(output_path, "steady_state_workload_final.svg"), schedule.mapping, schedule.ssis
        )


class AllocationArtifacts:
    """The ``allocation_artifacts`` observer: every allocation solved during the run writes its artifacts
    (see :mod:`stream.allocation.artifacts`) into ``<group output>/tetra``."""

    def __init__(self, *, run_name: str) -> None:
        self.run_name = run_name
        self._context = artifacts()
        self._context.__enter__()

    def instrument(self, stages: list[StageCallable]) -> list[StageCallable]:
        return stages

    def finish(self) -> None:
        self._context.__exit__(None, None, None)

    def fail(self, reason: str) -> None:
        self._context.__exit__(None, None, None)
