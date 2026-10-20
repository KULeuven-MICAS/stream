"""Energy in pJ: each node's on the cores it runs on, and each DMA stream's bits times the energy per bit of the memory
ports it reads and writes and of the links it crosses."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from stream.opt.allocation.constraint_optimization.families.traffic import (
    DmaStream,
    dma_streams,
    link_cores,
    link_name,
    placed_cores,
    port_name,
)
from stream.opt.allocation.constraint_optimization.utils import active_fraction

if TYPE_CHECKING:
    from stream.cost_model.core_cost import CoreCostEntry
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext

# (kind, resource, core ids): one row of the energy breakdown
Component = tuple[str, str, tuple[int, ...]]


def _energy_per_bit(stream: DmaStream) -> list[tuple[Component, float]]:
    """pJ per bit by where it is spent: each source's share of its read, each target's write, and every link the
    stream crosses."""
    ports = [
        (("port", port_name(side.port.key), side.port.core_ids), side.share * side.port.energy_per_bit(side.direction))
        for side in stream.sides
    ]
    links = stream.choice.links_used
    return ports + [
        (("link", link_name(link), link_cores(link)), link.unit_energy_cost / link.bandwidth) for link in links
    ]


class Energy:
    """Provides ``energy``, the pJ one iteration takes, and reports it; it adds no constraint, so it changes the
    allocation only where an objective level or bound names it."""

    name: ClassVar[str] = "energy"
    requires: ClassVar[tuple[str, ...]] = ("transfer_latency",)
    provides: ClassVar[tuple[str, ...]] = ("node_energy", "transfer_energy", "energy")

    def build(self, ctx: FormulationContext) -> None:
        q, space = ctx.quantities, ctx.space
        nodes = sum(
            space.cost_lut.get_cost(node, core).energy_total * active_fraction(node, space.ssis)
            for node in space.ssc_nodes
            for core in placed_cores(ctx, node)
        )
        transfers = [
            s.bits_per_cycle * sum(pj for _, pj in _energy_per_bit(s)) * s.active_cycles for s in dma_streams(ctx)
        ]
        q.add("node_energy", nodes)
        q.add("transfer_energy", ctx.model.quicksum(transfers)._raw if transfers else 0.0)
        q.add("energy", q.get("node_energy").expr + q.get("transfer_energy").expr)

    def report(self, ctx: FormulationContext) -> dict[str, Any]:
        """Per iteration, split into the nodes and the transfers and into the components it is spent in, and over
        every iteration of the run."""
        q, value = ctx.quantities, ctx.model.value
        nodes, transfers = value(q.get("node_energy").expr), value(q.get("transfer_energy").expr)
        return {
            "energy": {
                "node_pj_per_iteration": nodes,
                "transfer_pj_per_iteration": transfers,
                "per_iteration_pj": nodes + transfers,
                "total_pj": (nodes + transfers) * ctx.space.iterations,
                "components": _components(ctx),
            }
        }


def _components(ctx: FormulationContext) -> list[dict[str, Any]]:
    """The energy of one iteration by where it is spent, largest first: each node's MAC array and memory levels on
    the cores it runs on, and each transfer's memory ports and links."""
    pj: dict[Component, float] = defaultdict(float)
    for node in ctx.space.ssc_nodes:
        fraction = active_fraction(node, ctx.space.ssis)
        for core in placed_cores(ctx, node):
            for kind, resource, energy in _node_split(ctx.space.cost_lut.get_cost(node, core)):
                pj[(kind, resource, (core.id,))] += energy * fraction
    for stream in dma_streams(ctx):
        bits = ctx.model.value(stream.bits_per_cycle * stream.active_cycles)
        for component, pj_per_bit in _energy_per_bit(stream):
            pj[component] += bits * pj_per_bit
    rows = [
        {"kind": kind, "resource": resource, "core_ids": list(cores), "pj_per_iteration": energy}
        for (kind, resource, cores), energy in pj.items()
        if energy > 0
    ]
    return sorted(rows, key=lambda row: -row["pj_per_iteration"])


def _node_split(cost: CoreCostEntry) -> list[tuple[str, str, float]]:
    """A node's energy on one core: its MAC energy and each memory level's, when its cost model splits them."""
    cme = cost.cme
    if cme is None:
        return [("node", "node", cost.energy_total)]
    split = [("mac", "MAC", float(cme.mac_energy))]
    links = cme.layer.memory_operand_links
    for layer_op, per_level in cme.mem_energy_breakdown.items():
        levels = cme.accelerator.memory_hierarchy.get_memory_levels(links.layer_to_mem_op(layer_op))
        split += [
            ("memory", level.memory_instance.name, float(pj)) for level, pj in zip(levels, per_level, strict=True)
        ]
    return split
