"""Core-cost estimator backends, selected by hardware rather than hardcoded."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from stream.plugins import load_group
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator

if TYPE_CHECKING:
    from zigzag.mapping.temporal_mapping import TemporalMappingType

    from stream.cost_model.core_cost import CoreCostEntry
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core
    from stream.mapping.mapping import Mapping
    from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)

CORE_COST_BACKENDS_GROUP = "stream.core_cost_backends"


class CoreEstimator(Protocol):
    """What a backend produces: the object the stage calls once per node-core pair."""

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry: ...


class CoreCostContext(Protocol):
    """The subset of the estimation stage a backend reads to build its estimator."""

    workload: Workload
    accelerator: Accelerator
    mapping: Mapping
    temporal_mapping_type: TemporalMappingType
    loma_lpf_limit: int
    nb_spatial_mappings_generated: int


class CoreCostBackend(Protocol):
    """A discovered core-cost estimator backend. ``name`` becomes ``metadata["backend"]``; ``priority``
    breaks ties (highest wins, ZigZag lowest)."""

    name: str
    priority: int

    def claims(self, core: Core) -> bool:
        """Whether this backend models ``core``. Cheap predicate; no heavy imports."""
        ...

    def make(self, context: CoreCostContext) -> CoreEstimator:
        """Build the estimator, reading whatever it needs from the stage ``context``."""
        ...


class AIEBackend:
    """AIE compute tiles: the utilization-based estimator."""

    name = "aie"
    priority = 10

    def claims(self, core: Core) -> bool:
        return str(core.core_type).startswith("aie2.") and core.type == "compute"

    def make(self, context: CoreCostContext) -> CoreEstimator:
        from stream.stages.estimation.aie_cost_estimator import AIECostEstimator  # noqa: PLC0415

        return AIECostEstimator(context.workload, context.mapping)


class ZigZagBackend:
    """The universal fallback: claims every core at the lowest priority."""

    name = "zigzag"
    priority = 0

    def claims(self, core: Core) -> bool:  # noqa: ARG002 -- claims everything by design
        return True

    def make(self, context: CoreCostContext) -> CoreEstimator:
        return ZigZagCostEstimator(
            workload=context.workload,
            accelerator=context.accelerator,
            mapping=context.mapping,
            temporal_mapping_type=context.temporal_mapping_type,
            loma_lpf_limit=context.loma_lpf_limit,
            nb_spatial_mappings_generated=context.nb_spatial_mappings_generated,
        )


# Entry-point targets registered under the public distribution (see pyproject.toml).
AIE_BACKEND = AIEBackend()
ZIGZAG_BACKEND = ZigZagBackend()


def discover_backends() -> list[CoreCostBackend]:
    """Every registered core-cost backend, in discovery order (a later, higher-priority overlay
    registration comes last)."""
    backends: list[CoreCostBackend] = []
    for plugin in load_group(CORE_COST_BACKENDS_GROUP):
        obj = plugin.obj
        backends.append(obj() if isinstance(obj, type) else obj)
    return backends


def select_backend(core: Core, backends: list[CoreCostBackend] | None = None) -> CoreCostBackend:
    """The backend that costs ``core``: highest ``priority`` among those whose ``claims(core)`` is true.

    Ties go to the later registration (an overlay outranks a built-in of equal priority).
    """
    candidates = discover_backends() if backends is None else backends
    chosen: CoreCostBackend | None = None
    for backend in candidates:
        if backend.claims(core) and (chosen is None or backend.priority >= chosen.priority):
            chosen = backend
    if chosen is None:
        raise RuntimeError(
            f"no core-cost backend claims core {getattr(core, 'id', core)!r} (core_type "
            f"{getattr(core, 'core_type', '?')!r}); the built-in {CORE_COST_BACKENDS_GROUP!r} entry "
            "points are missing -- reinstall the package"
        )
    return chosen
