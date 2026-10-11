"""Energy in pJ: each node's on the cores it runs on, and each DMA stream's bits times the energy per bit of the memory
ports it reads and writes and of the links it crosses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from stream.opt.allocation.constraint_optimization.families.traffic import DmaStream, dma_streams, placed_cores
from stream.opt.allocation.constraint_optimization.utils import active_fraction

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext


def _energy_per_bit(stream: DmaStream) -> float:
    """pJ per bit: each source's share of its read, each target's write, and every link the stream crosses."""
    ports = sum(side.share * side.port.energy_per_bit(side.direction) for side in stream.sides)
    return ports + sum(link.unit_energy_cost / link.bandwidth for link in stream.choice.links_used)


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
        transfers = [s.bits_per_cycle * _energy_per_bit(s) * s.active_cycles for s in dma_streams(ctx)]
        q.add("node_energy", nodes)
        q.add("transfer_energy", ctx.model.quicksum(transfers)._raw if transfers else 0.0)
        q.add("energy", q.get("node_energy").expr + q.get("transfer_energy").expr)

    def report(self, ctx: FormulationContext) -> dict[str, Any]:
        """Per iteration, split into the nodes and the transfers, and over every iteration of the run."""
        q, value = ctx.quantities, ctx.model.value
        nodes, transfers = value(q.get("node_energy").expr), value(q.get("transfer_energy").expr)
        return {
            "energy": {
                "node_pj_per_iteration": nodes,
                "transfer_pj_per_iteration": transfers,
                "per_iteration_pj": nodes + transfers,
                "total_pj": (nodes + transfers) * ctx.space.iterations,
            }
        }
