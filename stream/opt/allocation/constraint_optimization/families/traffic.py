"""The bits each DMA stream and each node moves through the top-level memory ports, per iteration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.ports import READ, WRITE, Port, PortKey
from stream.opt.allocation.constraint_optimization.families.latency import copy_latencies, reuse_selectors
from stream.opt.allocation.constraint_optimization.utils import active_fraction, get_active_latency
from stream.stages.estimation.core_cost_backends import discover_backends, entry_backend, port_traffic
from stream.workload.demand import item_bits
from stream.workload.workload import ComputationNode, InEdge, TransferNode

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.noc.communication_link import CommunicationLink
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import CopySet
    from stream.opt.solver import SolverVar
    from stream.workload.tensor import Tensor


@dataclass(frozen=True)
class PortShare:
    """``share`` of a stream's bits pass ``port`` in ``direction`` at ``efficiency`` of the port's rate for the
    stream's access pattern."""

    port: Port
    direction: str
    share: float
    efficiency: float


@dataclass(frozen=True)
class DmaStream:
    """``bits_per_cycle * active_cycles`` is the stream's bits per iteration; ``active_cycles``, at most
    ``latency_ub``, is the transfer's latency on its chosen path; ``link_shares`` the share of the stream's bits
    each link carries."""

    choice: MulticastPathPlan
    active_cycles: Any
    bits_per_cycle: float
    latency_ub: float
    slot: int
    sides: tuple[PortShare, ...]
    energy_sides: tuple[PortShare, ...]
    link_shares: tuple[tuple[CommunicationLink, float], ...] = ()


@dataclass(frozen=True)
class NodeTraffic:
    """Bits per iteration a node moves through ``port`` of the core it runs on."""

    port: Port
    bits: float
    slot: int


def _sides(ctx: FormulationContext, tr: TransferNode, choice: Any, copies: CopySet | None = None) -> list[PortShare]:
    """The share of the transfer's bits each port of each memory on route ``choice`` passes, as
    :meth:`DecisionSpace.side_ports` gives it. A target passing data on reads what it holds of it, unless its cores'
    switch copies the data onto the links unread."""
    space = ctx.space
    sides: list[PortShare] = []
    runs = dict(zip((READ, WRITE), space.layouts.runs(tr, choice), strict=True))
    relays = tuple(
        memory
        for memory in choice.targets
        if memory not in choice.sources
        and choice.links_at(memory, leaving=True)
        and not space.hardware.circuit_switched(memory.cores)
    )
    for memories, direction in ((choice.sources + relays, READ), (choice.targets, WRITE)):
        for memory in dict.fromkeys(memories):
            for port, share in space.side_ports(tr, choice, memory, direction, copies).items():
                efficiency = port.bandwidth.efficiency(runs[direction], direction)
                sides.append(PortShare(port, direction, share, efficiency))
    return sides


def _walk_sides(
    ctx: FormulationContext, tr: TransferNode, choice: Any, sides: tuple[PortShare, ...]
) -> list[PortShare]:
    """``sides`` with each memory of a walked split placement at its share over the whole walk
    (:meth:`DecisionSpace.walk_shares`), even where the busiest iteration leaves it out: what energy is priced over."""
    space = ctx.space
    if not (shares := space.walk_shares(tr, choice)):
        return list(sides)
    names = {memory.name for memory in shares}
    found = [side for side in sides if side.port.key.memory not in names]
    reading = any(isinstance(node, InEdge) for node in space.workload.predecessors(tr))
    direction = READ if reading else WRITE
    for memory, share in shares.items():
        own = [side for side in sides if side.port.key.memory == memory.name]
        if not own:
            role = space.operand_role(tr, read_side=reading)
            ports = space.accelerator.ports.memory_ports(memory, direction, role, space.memory_links(memory, reading))
            own = [PortShare(port, direction, 1.0, 1.0) for port in ports]
        total = sum(side.share for side in own)
        found += [replace(side, share=share * side.share / total) for side in own if total]
    return found


def dma_streams(ctx: FormulationContext) -> list[DmaStream]:
    """Every transfer latency on a path choice that moves bits, with the ports its sources read and targets write.
    One per copy set where a route delivers to several, then the :func:`local_copies`."""
    q, space = ctx.quantities, ctx.space
    if "transfer_latency" not in q:
        return []
    streams: list[DmaStream] = []
    for (tr, choice), terms in copy_latencies(ctx).items():
        for copies, active in terms.items():
            link_latency = float(space.transfer_latency_for_path(tr, choice, copies))
            if link_latency <= 0:
                continue
            latency = get_active_latency(tr, link_latency, space.ssis)
            bits = space.moved_bits(tr, choice, copies=copies) * active_fraction(tr, space.ssis)
            sides = tuple(_sides(ctx, tr, choice, copies))
            energy_sides = tuple(_walk_sides(ctx, tr, choice, sides))
            shares = space.link_shares(tr, choice, copies)
            streams.append(
                DmaStream(choice, active, bits / latency, latency, space.slot_of[tr], sides, energy_sides, shares)
            )
    return streams + local_copies(ctx)


def local_copies(ctx: FormulationContext) -> list[DmaStream]:
    """Each local copy, per route: one read and one write of its box on that memory's
    ports per firing of the copy."""
    space = ctx.space
    streams: list[DmaStream] = []
    for (tr, choice), y in ctx.vars.y.items():
        bitwidth = tr.inputs[0].operand_type.bitwidth * active_fraction(tr, space.ssis)
        for k, item in enumerate(item for item in choice.demand if item.copied_from is not None):
            memory, copy = choice.targets[item.target], tr.outputs[item.copies[0]]
            bits = item_bits([item], bitwidth)
            role = space.operand_role(tr, read_side=False)
            ports = [(space.accelerator.ports.port_for(memory.cores[0], d, role, memory), d) for d in (READ, WRITE)]
            sides = tuple(PortShare(port, d, 1.0, 1.0) for port, d in ports if port is not None)
            local = MulticastPathPlan(sources=(memory,), targets=(memory,), total_hops_objective=0, links_used=())
            streams.append(
                DmaStream(
                    local,
                    _local_copy_bits(ctx, tr, choice, y, k, bits, copy),
                    1.0,
                    bits,
                    space.slot_of[tr],
                    sides,
                    sides,
                )
            )
    return streams


def _local_copy_bits(  # noqa: PLR0913
    ctx: FormulationContext,
    tr: TransferNode,
    choice: MulticastPathPlan,
    y: SolverVar,
    k: int,
    bits: float,
    copy: Tensor,
) -> Any:
    """Bits per iteration the ``k``-th local copy of ``tr`` moves while ``y`` chooses ``choice``: its bits over the
    copy's reuse, built once."""
    q = ctx.quantities
    if "local_copy_bits" in q and (tr, choice, k) in q.indexed("local_copy_bits"):
        return q.get("local_copy_bits", (tr, choice, k)).expr
    var = ctx.binary_times_const_over_linexpr(
        binary_var=y,
        numerator=bits,
        denominator_expr=q.get("copy_reuse_factor", copy).expr,
        denominator_lb=1.0,
        base_name=f"local_copy_{tr}_{k}",
        selectors=reuse_selectors(ctx, copy),
    )
    q.add("local_copy_bits", var._raw, index=(tr, choice, k))
    return var._raw


def placed_cores(ctx: FormulationContext, node: ComputationNode) -> list[Core]:
    """The cores ``node`` runs on, of those it has a cost on."""
    placed = {core for group in ctx.space.mapping.get(node).resource_allocation for core in group}
    return [core for core in ctx.space.cost_lut.get_cores(node) if core in placed]


def node_traffic(ctx: FormulationContext) -> list[NodeTraffic]:
    """The top-level port traffic of each node from its cost backend, on each core it runs on, at a port of the
    memory that core holds the operand in."""
    traffic: list[NodeTraffic] = []
    backends = discover_backends()
    accelerator = ctx.space.accelerator
    for node in ctx.space.ssc_nodes:
        fraction = active_fraction(node, ctx.space.ssis)
        for core in placed_cores(ctx, node):
            entry = ctx.space.cost_lut.get_cost(node, core)
            for operand, direction, bits in port_traffic(entry_backend(entry, core, backends), entry):
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
