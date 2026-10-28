"""The bits each DMA stream and each node moves through the top-level memory ports, per iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stream.hardware.ports import READ, WRITE, Port, PortKey
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
    ``latency_ub``, is the transfer's latency on its chosen path."""

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
    targets = range(len(choice.targets))
    return sum(ctx.space.moved_bits(tr, choice, target_at=j) for j in targets) / moved / len(targets)


def _sides(ctx: FormulationContext, tr: TransferNode, choice: Any) -> list[PortShare]:
    """The share of the transfer's bits each port of each memory on route ``choice`` passes. A memory's share sums
    over its positions (a broadcast reaches each memory once) and is split over its ports as in
    :meth:`DecisionSpace.side_ports`."""
    space = ctx.space
    sides: list[PortShare] = []
    runs = dict(zip((READ, WRITE), space.layouts.runs(tr, choice), strict=True))
    moved = space.moved_bits(tr, choice)
    broadcast = _target_share(ctx, tr, choice) == 1.0
    for memories, direction in ((choice.sources, READ), (choice.targets, WRITE)):
        for memory in dict.fromkeys(memories):
            positions = [i for i, m in enumerate(memories) if m is memory]
            if direction == WRITE and broadcast:
                positions = positions[:1]  # targets in one memory receive a broadcast once, into it
            at = "source_at" if direction == READ else "target_at"
            share = sum(space.moved_bits(tr, choice, **{at: position}) / moved for position in positions)
            for port, fraction in space.side_ports(tr, choice, memory, direction).items():
                efficiency = port.bandwidth.efficiency(runs[direction], direction)
                sides.append(PortShare(port, direction, share * fraction, efficiency))
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
        streams.append(DmaStream(choice, quantity.expr, bits / latency, latency, ctx.space.slot_of[tr], sides))
    return streams


def placed_cores(ctx: FormulationContext, node: ComputationNode) -> list[Core]:
    """The cores ``node`` runs on, of those it has a cost on."""
    placed = {core for group in ctx.space.mapping.get(node).resource_allocation for core in group}
    return [core for core in ctx.space.cost_lut.get_cores(node) if core in placed]


def node_traffic(ctx: FormulationContext) -> list[NodeTraffic]:
    """The top-level port traffic of each node from its cost backend, on each core it runs on, at a port of the
    memory that core holds the operand in."""
    traffic: list[NodeTraffic] = []
    backends: dict[Core, CoreCostBackend] = {}
    accelerator = ctx.space.accelerator
    for node in ctx.space.ssc_nodes:
        fraction = active_fraction(node, ctx.space.ssis)
        for core in placed_cores(ctx, node):
            backend = backends.setdefault(core, select_backend(core))
            for operand, direction, bits in port_traffic(backend, ctx.space.cost_lut.get_cost(node, core)):
                port = accelerator.ports.port_for(core, direction, operand, accelerator.memory_holding(core, operand))
                if port is not None:
                    traffic.append(NodeTraffic(port, bits * fraction, ctx.space.slot_of[node]))
    return traffic


def port_name(key: PortKey) -> str:
    """How reports name a memory port: ``<memory>.<port>``."""
    return f"{key.memory}.{key.port}"


def link_name(link: Any) -> str:
    """How reports name a link: ``bus <memories>`` for a bus, else ``<sender>-><receiver>``, with ``#<n>`` for the
    n-th of several between the same memories."""
    nth = f"#{link.parallel}" if link.parallel else ""
    if link.members:
        return f"bus{nth} " + ",".join(memory.name for memory in link.members)
    return f"{link.sender.name}->{link.receiver.name}{nth}"


def link_cores(link: Any) -> tuple[int, ...]:
    """The ids of the cores a link connects."""
    return tuple(core.id for core in link.cores)
