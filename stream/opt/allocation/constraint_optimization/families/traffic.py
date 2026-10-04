"""The bits each DMA stream and each node moves through the top-level memory ports, per iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stream.cost_model.bandwidth import contiguous_span_bytes
from stream.hardware.ports import OUTPUT, READ, WRITE, Port, input_role
from stream.opt.allocation.constraint_optimization.utils import active_fraction, get_active_latency
from stream.stages.estimation.core_cost_backends import CoreCostBackend, port_traffic, select_backend
from stream.workload.workload import ComputationNode, TransferNode

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext


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
    """``bits_per_cycle * active_cycles`` is the stream's bits per iteration; ``active_cycles``, at most
    ``latency_ub``, is the transfer's latency on its chosen path, or that path's choice when the latency rounds to 0."""

    choice: MulticastPathPlan
    active_cycles: Any
    bits_per_cycle: float
    latency_ub: float
    slot: int
    sides: tuple[PortShare, ...]


@dataclass(frozen=True)
class NodeTraffic:
    """Bits per iteration a node moves through ``port`` of the core it runs on."""

    port: Port
    bits: float
    slot: int


def transfer_operand_role(ctx: FormulationContext, tr: TransferNode, read_side: bool) -> str:
    """Operand role the tensor has on one side: the output for a producer, input k for a consumer's input k."""
    producer = next(iter(ctx.space.workload.predecessors(tr)), None)
    if read_side and isinstance(producer, ComputationNode):
        return OUTPUT
    for consumer in ctx.space.workload.successors(tr):
        if isinstance(consumer, ComputationNode):
            for tensor in tr.outputs:
                if tensor in consumer.inputs:
                    return input_role(consumer.inputs.index(tensor) + 1)
    return OUTPUT


def _span_bytes(tr: TransferNode) -> float:
    tensor = tr.inputs[0]
    full = tuple(tensor.subview.source.type.get_shape())
    return contiguous_span_bytes(tuple(tensor.shape), full, tensor.operand_type.bitwidth)


def _target_share(ctx: FormulationContext, tr: TransferNode) -> float:
    """Share of the transferred tensor one target receives: its tile under the consumer's inter-core tiling,
    as memory access estimation counts it. A broadcast gives every target the whole tensor."""
    tensor = tr.outputs[0]
    tile = ctx.space.workload.get_tensor_of_transfer_to_single_core(tensor, tr, ctx.space.mapping)
    return tile.size_bits() / tensor.size_bits()


def _sides(ctx: FormulationContext, tr: TransferNode, choice: Any) -> list[PortShare]:
    ports = ctx.space.accelerator.ports
    sides: list[PortShare] = []
    span = _span_bytes(tr)
    write_share = _target_share(ctx, tr)
    for cores, direction, share in (
        (choice.sources, READ, 1.0 / len(choice.sources)),
        (choice.targets, WRITE, write_share),
    ):
        operand = transfer_operand_role(ctx, tr, read_side=direction == READ)
        for core in cores:
            port = ports.port_for(core, direction, operand)
            if port is None:
                continue
            # Targets in one core_memory_sharing group receive a broadcast once, into their shared memory.
            broadcast_again = any(s.port.key == port.key and s.direction == WRITE for s in sides)
            if direction == WRITE and write_share == 1.0 and broadcast_again:
                continue
            sides.append(PortShare(port, direction, share, port.bandwidth.efficiency(span, direction)))
    return sides


def dma_streams(ctx: FormulationContext) -> list[DmaStream]:
    """Every transfer latency on a path choice that moves bits, with the ports its sources read and targets write."""
    streams: list[DmaStream] = []
    if "transfer_latency" not in ctx.quantities:
        return []
    for (tr, choice), quantity in ctx.quantities.indexed("transfer_latency").items():
        link_latency = float(ctx.space.transfer_latency_for_path(tr, choice))
        if link_latency <= 0:
            continue
        latency = get_active_latency(tr, link_latency, ctx.space.ssis)
        bits = tr.inputs[0].size_bits() * active_fraction(tr, ctx.space.ssis)
        sides = tuple(_sides(ctx, tr, choice))
        if latency > 0:
            stream = DmaStream(choice, quantity.expr, bits / latency, latency, ctx.space.slot_of[tr], sides)
        else:
            # Charged at a reuse factor of 1, an upper bound, since y / R needs a variable of its own.
            gate = ctx.vars.y[(tr, choice)]._raw
            stream = DmaStream(choice, gate, bits, 1.0, ctx.space.slot_of[tr], sides)
        streams.append(stream)
    return streams


def node_traffic(ctx: FormulationContext) -> list[NodeTraffic]:
    """The top-level port traffic of each node from its cost backend, on the physical core it runs on."""
    traffic: list[NodeTraffic] = []
    backends: dict[Core, CoreCostBackend] = {}
    for node in ctx.space.ssc_nodes:
        placed = {core for group in ctx.space.mapping.get(node).resource_allocation for core in group}
        fraction = active_fraction(node, ctx.space.ssis)
        for core in ctx.space.cost_lut.get_cores(node):
            if core not in placed:
                continue
            backend = backends.setdefault(core, select_backend(core))
            for operand, direction, bits in port_traffic(backend, ctx.space.cost_lut.get_cost(node, core)):
                port = ctx.space.accelerator.ports.port_for(core, direction, operand)
                if port is not None:
                    traffic.append(NodeTraffic(port, bits * fraction, ctx.space.slot_of[node]))
    return traffic
