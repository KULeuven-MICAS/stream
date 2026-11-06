"""The ``accelforge`` core-cost backend: AccelForge's single-Einsum mapper costs a node on a ZigZag compute core at the
extent and with the work per point Stream's ZigZag estimator costs it. At priority -10 it is chosen only by name, with
the ``core_cost_backend`` stage option; ``core_cost_options["accelforge"]`` gives it :class:`AccelForgeOptions`. See
``docs/source/accelforge.md``."""

from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import importlib.util
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xdsl.ir.affine import AffineDimExpr

from stream.cost_model.core_cost import CoreCostEntry
from stream.datatypes import ELEMENT_BITS
from stream.hardware.ports import OUTPUT, READ_BY_DATAPATH, WRITE_BY_DATAPATH, input_role
from stream.stages.estimation.accelforge.tree import COMPUTE, PJ, zigzag_core
from stream.stages.estimation.accelforge.workload import (
    ZIGZAG_INPUTS,
    UnsupportedConversion,
    describe,
    tensor_name,
)
from stream.stages.estimation.core_cost_backends import ZIGZAG_BACKEND
from stream.workload.utils import is_mac_operator_type, operand_inputs

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.stages.estimation.core_cost_backends import ContiguousAxes, CoreCostContext, PortTraffic
    from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
    from stream.workload.node import ComputationNode

logger = logging.getLogger(__name__)

NAME = "accelforge"
MODEL_REVISION = 1
"""Bumped whenever a change here or in the converters changes what an entry holds, so cached entries are recomputed."""


class AccelForgeEstimationError(RuntimeError):
    """AccelForge could not cost a node it can express on a core it claims."""


@dataclass(frozen=True)
class AccelForgeOptions:
    """``memory_order``: ``free`` (falling back to ``forced`` where AccelForge rejects its own template) or ``forced``;
    ``imperfect_spatial_loops``: let a fanout unroll part of a dim it does not divide."""

    memory_order: str = "free"
    imperfect_spatial_loops: bool = False

    @classmethod
    def parse(cls, options: Mapping[str, Any]) -> AccelForgeOptions:
        try:
            parsed = cls(**options)
        except TypeError as error:
            raise ValueError(f"core_cost_options[{NAME!r}]: {error}") from None
        if parsed.memory_order not in ("free", "forced"):
            raise ValueError(f"core_cost_options[{NAME!r}]: memory_order must be 'free' or 'forced'")
        return parsed


@dataclass(frozen=True)
class MappedNode:
    """AccelForge's best mapping of a node, in its units (cycles, J, actions), and how the search went."""

    latency: float
    energy_j: float
    actions: Mapping[tuple[str, str, str], float]
    """Action counts by (component, tensor, action); compute has tensor ``"None"``."""
    bits_per_action: Mapping[tuple[str, str], float]
    spatial_unrolling: Mapping[int, int]
    temporal_cycles: int
    top_storage: Mapping[str, str]
    """Per tensor, the outermost component storing it in the mapping."""
    units: int
    mapping: tuple[str, ...]
    memory_order: str
    fallback: str | None
    wall_s: float


@functools.cache
def accelforge_installed() -> bool:
    """Whether accelforge can be imported, found without importing it."""
    return importlib.util.find_spec("accelforge") is not None


class AccelForgeBackend:
    """Claims the ZigZag compute cores the arch converter accepts, when accelforge is installed."""

    name = NAME
    priority = -10

    def claims(self, core: Core) -> bool:
        if not accelforge_installed():
            return False
        try:
            zigzag_core(core)
        except UnsupportedConversion:
            return False
        return True

    def make(self, context: CoreCostContext) -> AccelForgeEstimator:
        options = AccelForgeOptions.parse(context.core_cost_options.get(NAME, {}))
        return AccelForgeEstimator(ZIGZAG_BACKEND.make(context), options)

    def cache_identity(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """The options, the converters' revision and AccelForge's version."""
        version = importlib.metadata.version(NAME) if accelforge_installed() else None
        return {**dataclasses.asdict(AccelForgeOptions.parse(options)), "model_revision": MODEL_REVISION, NAME: version}

    def port_traffic(self, entry: CoreCostEntry) -> PortTraffic:
        return tuple(entry.metadata.get("port_traffic", ()))

    def contiguous_axes(self, entry: CoreCostEntry) -> ContiguousAxes:
        """Per operand, the axes AccelForge's spatial loops unroll widest over the array, as ZigZag's rule reads its
        own spatial mapping."""
        unrolled: dict[int, int] = entry.metadata.get("spatial_unrolling", {})
        if entry.layer is None:
            return ()
        needs: list[frozenset[int]] = []
        for access in entry.layer.operand_mapping:
            factors = {
                axis: unrolled.get(result.position, 1)
                for axis, result in enumerate(access.results)
                if isinstance(result, AffineDimExpr)
            }
            widest = max(factors.values(), default=1)
            needs.append(frozenset(axis for axis, f in factors.items() if f == widest and f > 1))
        return tuple(needs)


@dataclass
class AccelForgeEstimator:
    zigzag: ZigZagCostEstimator
    """The run's ZigZag estimator: it sizes the nodes and costs those no Einsum can express."""
    options: AccelForgeOptions

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry:
        """AccelForge's cost of ``node`` on ``core``; ZigZag's, recording why, for a node no Einsum can express."""
        try:
            einsum = describe(node, tuple(self.zigzag.create_layer_dim_sizes(node).data.values()))
        except UnsupportedConversion as refusal:
            logger.warning(
                "AccelForge cannot express %s, so ZigZag costs it on core %s: %s", node.name, core.id, refusal
            )
            entry = self.zigzag.estimate(node, core)
            return dataclasses.replace(entry, metadata={**entry.metadata, "accelforge_refused": str(refusal)})
        from stream.stages.estimation.accelforge.mapper import map_node  # noqa: PLC0415 -- accelforge is optional

        try:
            mapped = map_node(einsum, core, self.options.memory_order, self.options.imperfect_spatial_loops)
        except Exception as error:
            raise AccelForgeEstimationError(
                f"AccelForge could not cost node {node.name} on core {core.id} ({core.core_type}): "
                f"{type(error).__name__}: {error}"
            ) from error
        return self.entry(node, core, mapped)

    def entry(self, node: ComputationNode, core: Core, mapped: MappedNode) -> CoreCostEntry:
        """The entry of ``mapped``, with each point of ``node`` costing the operations and cycles of the operators it
        expands into, as Stream's ZigZag estimator charges them."""
        ops, cycles_per_point = self.zigzag.work_per_op(node)
        macs = mapped.actions.get((COMPUTE, "None", "compute"), 0.0)
        compute = mapped.temporal_cycles * cycles_per_point
        metadata = {
            "backend": NAME,
            "mapping": mapped.mapping,
            "mapper_wall_s": mapped.wall_s,
            "memory_order": mapped.memory_order,
            "spatial_unrolling": dict(mapped.spatial_unrolling),
            "port_traffic": port_traffic(node, core, mapped),
        }
        if mapped.fallback:
            metadata["memory_order_fallback"] = mapped.fallback
        return CoreCostEntry(
            energy_total=mapped.energy_j / PJ + macs * core.operational_array.unit.energy_cost * (ops - 1),
            latency_total=max(mapped.latency, compute),
            ideal_cycle=float(math.ceil(math.ceil(macs / mapped.units) * cycles_per_point)),
            ideal_temporal_cycle=float(compute),
            layer=node,
            metadata=metadata,
        )


def port_traffic(node: ComputationNode, core: Core, mapped: MappedNode) -> PortTraffic:
    """Bits each operand a ZigZag core serves moves between its top memory, its outermost storage in the mapping,
    and the datapath below. Partial sums, which AccelForge keeps there at the output's precision, move at the core's
    accumulator precision, as ZigZag prices them."""
    operands = [(tensor, input_role(k)) for k, tensor in enumerate(operand_inputs(node)[:ZIGZAG_INPUTS], start=1)]
    operands.append((node.outputs[0], OUTPUT))
    accumulator = (getattr(core, "operand_precision", None) or {}).get("accumulator")
    traffic: list[tuple[str, str, float]] = []
    for tensor, role in operands:
        name = tensor_name(tensor)
        if (top := mapped.top_storage.get(name)) is None:
            continue
        reads, writes = (
            mapped.actions.get((top, name, a), 0.0) * mapped.bits_per_action[top, a] for a in ("read", "write")
        )
        if role == OUTPUT and accumulator and is_mac_operator_type(node.type) and reads:
            ratio = ELEMENT_BITS[accumulator] / tensor.operand_type.bitwidth
            # Every partial sum read back was written by an earlier pass: those writes, not the last pass's, widen too
            reads, writes = reads * ratio, writes + min(reads, writes) * (ratio - 1)
        traffic.extend(
            (role, direction, bits)
            for direction, bits in ((READ_BY_DATAPATH, reads), (WRITE_BY_DATAPATH, writes))
            if bits
        )
    return tuple(traffic)


ACCELFORGE_BACKEND = AccelForgeBackend()
