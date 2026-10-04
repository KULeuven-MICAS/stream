"""Core-cost estimator backends, selected by hardware rather than hardcoded."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from zigzag.hardware.architecture.memory_port import DataDirection

from stream.hardware.architecture.backends.zigzag import ZIGZAG_DIRECTION_NAMES, operand_role
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
CONTRACT_VERSION = 1

PortTraffic = tuple[tuple[str, str, float], ...]


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
    fusion_splits: dict


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


@runtime_checkable
class PortTrafficSource(Protocol):
    """Optional backend capability: the bits a costed node moves through its core's top-level memory ports."""

    def port_traffic(self, entry: CoreCostEntry) -> PortTraffic:
        """(operand role, direction, bits) per evaluation of ``entry``'s node, before its active fraction."""
        ...


def port_traffic(backend: object, entry: CoreCostEntry) -> PortTraffic:
    """``backend``'s port traffic for ``entry``; none from a backend without the capability."""
    return backend.port_traffic(entry) if isinstance(backend, PortTrafficSource) else ()


class AIEBackend:
    """AIE compute tiles: the kernel-library-priced estimator."""

    name = "aie"
    priority = 10

    def claims(self, core: Core) -> bool:
        return str(core.core_type).startswith("aie2.") and core.type == "compute"

    def make(self, context: CoreCostContext) -> CoreEstimator:
        from stream.stages.estimation.aie_cost_estimator import AIECostEstimator  # noqa: PLC0415

        return AIECostEstimator(context.workload, context.mapping, context.fusion_splits)


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

    def port_traffic(self, entry: CoreCostEntry) -> PortTraffic:
        """Words each operand's top level moves to and from the datapath, times the evaluated port's width."""
        cme = entry.cme
        if cme is None:
            return ()
        traffic: list[tuple[str, str, float]] = []
        for layer_op in cme.layer.layer_operands:
            mem_op = cme.memory_operand_links.layer_to_mem_op(layer_op)
            top = cme.mapping.mem_level[layer_op] - 1
            level = cme.accelerator.get_memory_level(mem_op, top)
            accesses = cme.memory_word_access[layer_op][top]
            for direction in (DataDirection.RD_OUT_TO_LOW, DataDirection.WR_IN_BY_LOW):
                words = accesses.get(direction)
                port = next((p for p in level.ports if (mem_op, top, direction) in p.served_op_lv_dir), None)
                if words and port is not None:
                    traffic.append((operand_role(str(mem_op)), ZIGZAG_DIRECTION_NAMES[direction], words * port.bw_max))
        return tuple(traffic)


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
