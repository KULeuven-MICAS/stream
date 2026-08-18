"""Core-cost estimator backends, selected by hardware rather than hardcoded.

A core-cost backend models how long (and how much energy) a computation node takes on one core. Which
backend costs a given core is a plugin decision: the engine ships the AIE and ZigZag backends as
ordinary entry points under the public distribution, and an out-of-tree overlay registers a backend
for its own hardware namespace without editing this file. The dispatch is "highest-priority backend
whose ``claims(core)`` is true", so the ZigZag backend claiming everything at the lowest priority is
the universal fallback -- a run with no overlay behaves exactly as the old ``if is_aie_compute_core``
branch did.

A backend advertises itself cheaply through :class:`CoreCostBackend` (``name``, ``priority``,
``claims``) and only builds the actual estimator on demand via ``make`` -- so a backend that pulls a
heavy optional toolchain (the AIE one) claims without importing it, and imports only when it is the
one selected.
"""

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

# Bumped when this seam's Protocol changes shape. A backend declares the version it was written
# against so a mismatch is a named error, not an AttributeError three frames deep.
CONTRACT_VERSION = 1


class CoreEstimator(Protocol):
    """What a backend produces: the object the stage calls once per node-core pair."""

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry: ...


class CoreCostContext(Protocol):
    """The subset of the estimation stage a backend reads to build its estimator.

    The stage satisfies this structurally, so a backend preserves its own constructor args (ZigZag
    takes more than AIE) without the seam depending on the concrete stage type.
    """

    workload: Workload
    accelerator: Accelerator
    mapping: Mapping
    temporal_mapping_type: TemporalMappingType
    loma_lpf_limit: int
    nb_spatial_mappings_generated: int


class CoreCostBackend(Protocol):
    """A core-cost estimator backend, discovered rather than hardcoded.

    ``name`` becomes ``metadata["backend"]`` on the produced :class:`CoreCostEntry` (provenance).
    ``priority`` breaks ties when several backends claim the same core -- highest wins, so the
    universal ZigZag fallback sits at the lowest. ``claims`` must be cheap and import nothing heavy.
    """

    name: str
    priority: int

    def claims(self, core: Core) -> bool:
        """Whether this backend models ``core``. Cheap predicate; no heavy imports."""
        ...

    def make(self, context: CoreCostContext) -> CoreEstimator:
        """Build the estimator, reading whatever it needs from the stage ``context``."""
        ...


class AIEBackend:
    """AIE compute tiles: the utilization-based estimator. Claims cheaply, imports lazily.

    The AIE estimator pulls in the AIE toolchain (an optional install), so the import stays inside
    ``make`` -- only a selected AIE backend triggers it, keeping the base path import-clean.
    """

    name = "aie"
    priority = 10

    def claims(self, core: Core) -> bool:
        return str(core.core_type).startswith("aie2.") and core.type == "compute"

    def make(self, context: CoreCostContext) -> CoreEstimator:
        from stream.stages.estimation.aie_cost_estimator import AIECostEstimator  # noqa: PLC0415

        return AIECostEstimator(context.workload, context.mapping)


class ZigZagBackend:
    """The universal fallback: claims every core at the lowest priority. Its internal
    ZigZag-to-ideal-cycle fallback is untouched -- it stays inside the ZigZag estimator."""

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


# Registered as entry points under the public distribution (see pyproject.toml). The instances are the
# entry-point targets, so the built-in default travels the same discovery path an overlay does.
AIE_BACKEND = AIEBackend()
ZIGZAG_BACKEND = ZigZagBackend()


def discover_backends() -> list[CoreCostBackend]:
    """Every registered core-cost backend, in discovery order (a later, higher-priority overlay
    registration comes last). This is the single ``load_group`` site for this seam."""
    backends: list[CoreCostBackend] = []
    for plugin in load_group(CORE_COST_BACKENDS_GROUP):
        obj = plugin.obj
        backends.append(obj() if isinstance(obj, type) else obj)
    return backends


def select_backend(core: Core, backends: list[CoreCostBackend] | None = None) -> CoreCostBackend:
    """The backend that costs ``core``: highest ``priority`` among those whose ``claims(core)`` is true.

    Ties go to the later registration (an overlay outranks a built-in of equal priority), matching the
    precedence ``load_group`` already applies across distributions.
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
