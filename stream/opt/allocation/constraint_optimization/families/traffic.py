"""The bits each DMA stream and each node moves through the top-level memory ports, per iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stream.hardware.ports import READ, WRITE, Port
from stream.opt.allocation.constraint_optimization.utils import active_fraction, get_active_latency
from stream.stages.estimation.core_cost_backends import CoreCostBackend, port_traffic, select_backend
from stream.workload.workload import TransferNode

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


def _target_share(ctx: FormulationContext, tr: TransferNode, choice: Any = None) -> float:
    """Share of the bits a transfer moves one target receives, on average over the targets of ``choice`` (its first
    route by default): 1 for a broadcast, every target receiving the whole tensor."""
    choice = choice or ctx.space.path_choices[tr][0]
    moved = ctx.space.moved_bits(tr, choice)
    return sum(ctx.space.moved_bits(tr, choice, target=t) for t in choice.targets) / moved / len(choice.targets)


def _sides(ctx: FormulationContext, tr: TransferNode, choice: Any) -> list[PortShare]:
    ports = ctx.space.accelerator.ports
    sides: list[PortShare] = []
    runs = dict(zip((READ, WRITE), ctx.space.layouts.runs(tr, choice), strict=True))
    moved = ctx.space.moved_bits(tr, choice)
    broadcast = _target_share(ctx, tr, choice) == 1.0
    for cores, direction in ((choice.sources, READ), (choice.targets, WRITE)):
        operand = ctx.space.operand_role(tr, read_side=direction == READ)
        for core in cores:
            side = {"source": core} if direction == READ else {"target": core}
            share = ctx.space.moved_bits(tr, choice, **side) / moved
            port = ports.port_for(core, direction, operand)
            if port is None:
                continue
            # Targets in one core_memory_sharing group receive a broadcast once, into their shared memory.
            broadcast_again = any(s.port.key == port.key and s.direction == WRITE for s in sides)
            if direction == WRITE and broadcast and broadcast_again:
                continue
            sides.append(PortShare(port, direction, share, port.bandwidth.efficiency(runs[direction], direction)))
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
        bits = ctx.space.moved_bits(tr, choice) * active_fraction(tr, ctx.space.ssis)
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
