"""The bits crossing the off-chip boundary each iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families import LATENCY, OFFCHIP_TRAFFIC
from stream.opt.solver import ObjectiveLevel

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry


class OffchipTraffic:
    """The bits crossing the off-chip boundary, the objective level after the latency, and charged in the latency
    at the time the off-chip links take to move them; nothing is charged where a shared-bandwidth model already
    times them. Slot latency only sees a transfer when it is the longest thing in its slot, so a transfer hiding
    behind compute is free to the latency however often it fires, but every slot shares the off-chip bandwidth."""

    name: ClassVar[str] = "offchip_traffic"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("offchip_traffic_weight",)

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        if not ctx.space.shared_bandwidth and (bandwidth := ctx.space.offchip_bandwidth()):
            q.add("offchip_traffic_weight", ctx.space.iterations / bandwidth)

    def objective(self, ctx: FormulationContext, q: QuantityRegistry) -> list[ObjectiveLevel]:
        space, z_stop = ctx.space, ctx.vars.z_stop
        traffic = ctx.model.quicksum(
            t.size_bits() / space.reuse_levels[(t, s)] * z_stop[(t, s)]._raw
            for t in space.tensors_to_optimize_reuse_for
            for s in space.stops(t)
        )._raw
        levels = [ObjectiveLevel(expr=traffic, priority=OFFCHIP_TRAFFIC, name="offchip_traffic")]
        if "offchip_traffic_weight" in q:
            levels.append(
                ObjectiveLevel(expr=q.get("offchip_traffic_weight").expr * traffic, priority=LATENCY, name="latency")
            )
        return levels
