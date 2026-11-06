"""Core-cost estimator backends, selected by hardware rather than hardcoded."""

from __future__ import annotations

import logging
from collections import abc
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from xdsl.ir.affine import AffineDimExpr
from zigzag.hardware.architecture.memory_port import DataDirection

from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND
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
COST_BACKEND = "cost_backend"
"""``metadata`` key of the backend a run chose for an entry's core and that backend's cache identity."""

PortTraffic = tuple[tuple[str, str, float], ...]
ContiguousAxes = tuple[frozenset[int], ...]


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
    core_cost_options: abc.Mapping[str, abc.Mapping[str, Any]]
    """Per backend name, the options the run gives that backend (the ``core_cost_options`` stage option)."""


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
class ConfigurableBackend(Protocol):
    """Optional backend capability: it takes options, and says what besides a node and core decides its entries."""

    def cache_identity(self, options: abc.Mapping[str, Any]) -> Any:
        """What decides an entry priced with ``options``, such as the options and the model's version. Raises
        ``ValueError`` for an option it does not take."""
        ...


def cache_identity(backend: CoreCostBackend, options: abc.Mapping[str, Any]) -> Any:
    """``backend``'s cache identity with ``options``; None for a backend that takes no options."""
    if isinstance(backend, ConfigurableBackend):
        return backend.cache_identity(options)
    if options:
        raise ValueError(f"the {backend.name!r} core-cost backend takes no options")
    return None


@runtime_checkable
class PortTrafficSource(Protocol):
    """Optional backend capability: the bits a costed node moves through its core's top-level memory ports."""

    def port_traffic(self, entry: CoreCostEntry) -> PortTraffic:
        """(operand role, direction, bits) per evaluation of ``entry``'s node, before its active fraction."""
        ...


def port_traffic(backend: object, entry: CoreCostEntry) -> PortTraffic:
    """``backend``'s port traffic for ``entry``; none from a backend without the capability."""
    return backend.port_traffic(entry) if isinstance(backend, PortTrafficSource) else ()


@runtime_checkable
class OperandLayoutSource(Protocol):
    """Optional backend capability: what layout a costed node's kernel needs its operands in."""

    def contiguous_axes(self, entry: CoreCostEntry) -> ContiguousAxes:
        """Per tensor of ``entry``'s node (its inputs, then its outputs), the axes its core reads or writes together
        in one access, which a copy of the tensor feeding it at full rate has innermost, in any order."""
        ...


def contiguous_axes(backend: object, entry: CoreCostEntry) -> ContiguousAxes:
    """``backend``'s operand layout needs for ``entry``; none from a backend without the capability."""
    return backend.contiguous_axes(entry) if isinstance(backend, OperandLayoutSource) else ()


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

    def contiguous_axes(self, entry: CoreCostEntry) -> ContiguousAxes:
        """Per operand, the axes its spatial mapping unrolls widest over the array: one memory word feeds the units
        along them, so they are laid out innermost (the channels of a convolution on an array unrolling them, a
        matmul's contraction along the rows of a systolic array, and both axes of the block a stationary operand
        loads). An axis unrolled less, such as one shared by a pair of rows, is fed word by word and needs nothing."""
        cme, node = entry.cme, entry.layer
        if cme is None or node is None:
            return ()
        unrolled: dict[str, float] = {}
        for single in cme.layer.spatial_mapping.values():
            for layer_dim, factor in single.items():
                unrolled[layer_dim.name] = unrolled.get(layer_dim.name, 1) * factor
        needs: list[frozenset[int]] = []
        for access in node.operand_mapping:
            factors = {
                axis: unrolled.get(f"D{result.position}", 1)
                for axis, result in enumerate(access.results)
                if isinstance(result, AffineDimExpr)
            }
            widest = max(factors.values(), default=1)
            needs.append(frozenset(axis for axis, f in factors.items() if f == widest and f > 1))
        return tuple(needs)

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


def select_backend(
    core: Core, backends: list[CoreCostBackend] | None = None, prefer: str | None = None
) -> CoreCostBackend:
    """The backend that costs ``core``: the one named ``prefer`` if it claims ``core``, else the highest ``priority``
    among those whose ``claims(core)`` is true, ties going to the later registration (an overlay outranks a built-in of
    equal priority)."""
    candidates = discover_backends() if backends is None else backends
    if prefer is not None:
        preferred = _named(prefer, candidates)
        if preferred is None:
            raise ValueError(f"unknown core-cost backend {prefer!r}; {_registered(candidates)}")
        if preferred.claims(core):
            return preferred
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


def entry_backend(
    entry: CoreCostEntry, core: Core, backends: list[CoreCostBackend] | None = None
) -> CoreCostBackend | None:
    """The backend that priced ``entry``: the one its ``metadata["backend"]`` names (ZigZag for its ideal-cycle
    fallback), the priority rule's choice for ``core`` when the entry names none, and None for a backend no longer
    registered."""
    candidates = discover_backends() if backends is None else backends
    tag = entry.metadata.get("backend")
    if tag is None:
        return select_backend(core, candidates)
    return _named(ZIGZAG_BACKEND.name if tag == IDEAL_CYCLE_BACKEND else tag, candidates)


def _named(name: str, backends: list[CoreCostBackend]) -> CoreCostBackend | None:
    """The backend called ``name`` registered last, as a later registration wins the priority rule's ties."""
    return next((b for b in reversed(backends) if b.name == name), None)


def _registered(backends: list[CoreCostBackend]) -> str:
    return "registered: " + ", ".join(sorted({b.name for b in backends}))
