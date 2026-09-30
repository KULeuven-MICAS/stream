"""The bits each DMA stream and each node moves through the top-level memory ports, per iteration."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

from stream.cost_model.bandwidth import contiguous_span_bytes
from stream.hardware.ports import OUTPUT, READ, WRITE, Port, input_role
from stream.opt.allocation.constraint_optimization.utils import get_active_latency
from stream.stages.estimation.core_cost_backends import CoreCostBackend, port_traffic, select_backend
from stream.workload.steady_state.iteration_space import LoopEffect
from stream.workload.workload import ComputationNode, TransferNode

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )


@dataclass(frozen=True)
class PortShare:
    """``share`` of a stream's bits pass ``port`` in ``direction`` (1/|sources| on a read, 1 on a write) at
    ``efficiency`` of the port's rate for the stream's access pattern."""

    port: Port
    direction: str
    share: float
    efficiency: float


@dataclass(frozen=True)
class DmaStream:
    """``bits_per_cycle * gated`` is the stream's bits per iteration; ``gated`` never exceeds ``latency_ub``."""

    transfer: TransferNode
    choice: MulticastPathPlan
    gated: Any
    bits_per_cycle: float
    latency_ub: float
    slot: int
    sides: tuple[PortShare, ...]


@dataclass(frozen=True)
class NodeTraffic:
    """Bits per iteration ``node`` moves through ``port`` of ``core``, the core it runs on."""

    node: ComputationNode
    core: Core
    port: Port
    direction: str
    bits: float
    slot: int


def active_fraction(node: Any, alloc: TransferAndTensorAllocator) -> float:
    """Fraction of the steady-state iterations in which ``node`` is not idle on an absent loop."""
    temporal = alloc.ssis.get(node).get_temporal_variables()
    total = prod(v.size for v in temporal)
    return prod(v.size for v in temporal if v.effect != LoopEffect.ABSENT) / total if total else 1.0


def operand_role(alloc: TransferAndTensorAllocator, tr: TransferNode, read_side: bool) -> str:
    """Operand role the tensor has on one side: the output for a producer, input k for a consumer's input k."""
    producer = next(iter(alloc.workload.predecessors(tr)), None)
    if read_side and isinstance(producer, ComputationNode):
        return OUTPUT
    for consumer in alloc.workload.successors(tr):
        if isinstance(consumer, ComputationNode):
            for tensor in tr.outputs:
                if tensor in consumer.inputs:
                    return input_role(consumer.inputs.index(tensor) + 1)
    return OUTPUT


def _span_bytes(tr: TransferNode) -> float:
    tensor = tr.inputs[0]
    full = tuple(tensor.subview.source.type.get_shape())
    return contiguous_span_bytes(tuple(tensor.shape), full, tensor.operand_type.bitwidth)


def _sides(alloc: TransferAndTensorAllocator, tr: TransferNode, choice: Any) -> list[PortShare]:
    ports = alloc.accelerator.ports
    sides: list[PortShare] = []
    span = _span_bytes(tr)
    for cores, direction, share in ((choice.sources, READ, 1.0 / len(choice.sources)), (choice.targets, WRITE, 1.0)):
        operand = operand_role(alloc, tr, read_side=direction == READ)
        for core in cores:
            port = ports.port_for(core, direction, operand)
            if port is None:
                continue
            # Targets in one core_memory_sharing group receive a multicast once, into their shared memory.
            if direction == WRITE and any(s.port.key == port.key and s.direction == WRITE for s in sides):
                continue
            sides.append(PortShare(port, direction, share, port.bandwidth.efficiency(span, direction)))
    return sides


def dma_streams(alloc: TransferAndTensorAllocator) -> list[DmaStream]:
    """Every gated transfer latency that moves bits, with the ports its sources read and its targets write."""
    streams: list[DmaStream] = []
    for (tr, choice), quantity in alloc.quantities.indexed("transfer_latency").items():
        latency = get_active_latency(tr, float(alloc.transfer_latency_for_path(tr, choice)), alloc.ssis)
        if latency <= 0:
            continue
        bits = tr.inputs[0].size_bits() * active_fraction(tr, alloc)
        sides = tuple(_sides(alloc, tr, choice))
        streams.append(DmaStream(tr, choice, quantity.expr, bits / latency, latency, alloc.slot_of[tr], sides))
    return streams


def node_traffic(alloc: TransferAndTensorAllocator) -> list[NodeTraffic]:
    """The top-level port traffic of each node from its cost backend, on the physical core it runs on."""
    traffic: list[NodeTraffic] = []
    backends: dict[Core, CoreCostBackend] = {}
    for node in alloc.ssc_nodes:
        placed = {core for group in alloc.mapping.get(node).resource_allocation for core in group}
        fraction = active_fraction(node, alloc)
        for core in alloc.cost_lut.get_cores(node):
            if core not in placed:
                continue
            backend = backends.setdefault(core, select_backend(core))
            for operand, direction, bits in port_traffic(backend, alloc.cost_lut.get_cost(node, core)):
                port = alloc.accelerator.ports.port_for(core, direction, operand)
                if port is not None:
                    traffic.append(NodeTraffic(node, core, port, direction, bits * fraction, alloc.slot_of[node]))
    return traffic
