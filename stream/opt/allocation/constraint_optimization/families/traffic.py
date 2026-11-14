"""The bits each DMA stream and each node moves through the top-level memory ports, per iteration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.ports import OUTPUT, READ, READ_BY_DATAPATH, WRITE, WRITE_BY_DATAPATH, Port, PortKey
from stream.opt.allocation.constraint_optimization.families.latency import copy_latencies, reuse_selectors
from stream.opt.allocation.constraint_optimization.utils import absent_trips, active_fraction, get_active_latency
from stream.stages.estimation.core_cost_backends import discover_backends, entry_backend, port_traffic
from stream.workload.demand import item_bits
from stream.workload.workload import ComputationNode, InEdge, TransferNode

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    transfer: TransferNode | None = None
    copies: CopySet | None = None


@dataclass(frozen=True)
class NodeTraffic:
    """Bits per iteration a node moves through ``port`` of the core it runs on."""

    port: Port
    bits: float
    slot: int
    node: ComputationNode


def _sides(ctx: FormulationContext, tr: TransferNode, choice: Any, copies: CopySet | None = None) -> list[PortShare]:
    """The share of the transfer's bits each port of each memory on route ``choice`` passes, as
    :meth:`DecisionSpace.side_ports` gives it. A target passing data on reads what it holds of it, unless its cores'
    switch copies the data onto the links unread. A memory a partial sum is added into is read and written for each add;
    the add itself is not costed."""
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
    summed_in = choice.sum_memories()
    for memories, direction in ((choice.sources + relays + summed_in, READ), (choice.targets + summed_in, WRITE)):
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
                DmaStream(
                    choice,
                    active,
                    bits / latency,
                    latency,
                    space.slot_of[tr],
                    sides,
                    energy_sides,
                    shares,
                    transfer=tr,
                    copies=copies,
                )
            )
    return streams + local_copies(ctx)


def firing_port_cycles(ctx: FormulationContext, tr: TransferNode, choice: MulticastPathPlan) -> float:
    """Cycles the busiest top-level port spends on one whole firing of ``tr`` on ``choice``."""
    space = ctx.space
    if space.transfer_latency_for_path(tr, choice) <= 0:
        return 0.0
    bits = space.moved_bits(tr, choice)
    per_port: dict[PortKey, float] = defaultdict(float)
    for side in _sides(ctx, tr, choice):
        per_port[side.port.key] += bits * side.share / side.efficiency / side.port.bits_per_cycle
    return max(per_port.values(), default=0.0)


def call_port_cycles(ctx: FormulationContext) -> dict[ComputationNode, float]:
    """Per node, the cycles the busiest top-level port spends on one whole call of it (:func:`node_traffic`)."""
    per_port: dict[tuple[ComputationNode, PortKey], float] = defaultdict(float)
    for traffic in node_traffic(ctx):
        calls = absent_trips(ctx.space.ssis[traffic.node])
        per_port[(traffic.node, traffic.port.key)] += traffic.bits * calls / traffic.port.bits_per_cycle
    busiest: dict[ComputationNode, float] = defaultdict(float)
    for (node, _), cycles in per_port.items():
        busiest[node] = max(busiest[node], cycles)
    return dict(busiest)


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
                    transfer=tr,
                    copies=item.copies,
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
    """The top-level port traffic of each node from its cost backend and of its running sums, on each core it runs
    on, at a port of the memory that core holds the operand in."""
    traffic: list[NodeTraffic] = []
    backends = discover_backends()
    for node in ctx.space.ssc_nodes:
        fraction = active_fraction(node, ctx.space.ssis)
        for core in placed_cores(ctx, node):
            entry = ctx.space.cost_lut.get_cost(node, core)
            charged = port_traffic(entry_backend(entry, core, backends), entry)
            for port, _, bits in [*_at_ports(ctx, core, charged), *running_sum_ports(ctx, node, core)]:
                traffic.append(NodeTraffic(port, bits * fraction, ctx.space.slot_of[node], node))
    return traffic


def running_sum_ports(ctx: FormulationContext, node: ComputationNode, core: Core) -> list[tuple[Port, str, float]]:
    """Each port of ``core`` a firing of ``node`` passes its running sums through, with the direction and the bits."""
    return _at_ports(ctx, core, _running_sum_traffic(ctx, node))


def _at_ports(
    ctx: FormulationContext, core: Core, charged: Iterable[tuple[str, str, float]]
) -> list[tuple[Port, str, float]]:
    """Each operand's ``(operand, direction, bits)`` of ``charged`` at the port of the memory ``core`` holds it in."""
    accelerator = ctx.space.accelerator
    found = []
    for operand, direction, bits in charged:
        port = accelerator.ports.port_for(core, direction, operand, accelerator.memory_holding(core, operand))
        if port is not None:
            found.append((port, direction, bits))
    return found


def _running_sum_traffic(ctx: FormulationContext, node: ComputationNode) -> list[tuple[str, str, float]]:
    """What a firing of ``node`` moves of each running sum beyond what its cost backend counts: the extra bits of
    its writes at the accumulator precision, and the read-back of the sum on every firing but the first."""
    space = ctx.space
    found: list[tuple[str, str, float]] = []
    for t in node.outputs:
        elements = space.workload.get_tensor_single_core(t, node, space.mapping).size_elements()
        sums = space.reduction_trips(t)
        if (bits := space.running_sum_bits(t)) is not None:
            partial = sums - 1 if space.sums_on_one_core(t) else sums
            found.append((OUTPUT, WRITE_BY_DATAPATH, elements * (bits - t.operand_type.bitwidth) * partial / sums))
        if sums > 1:
            found.append((OUTPUT, READ_BY_DATAPATH, elements * space.element_bits(t) * (sums - 1) / sums))
    return found


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
