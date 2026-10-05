"""The bits crossing the off-chip boundary each iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.families import LATENCY, OFFCHIP_TRAFFIC
from stream.opt.solver import ObjectiveLevel

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext


class OffchipTraffic:
    """The bits crossing the off-chip boundary, the objective level after the latency; with ``charge`` the latency
    also pays the time the off-chip links take to move them, which a slot hides when compute is longer, unless a
    shared-bandwidth model already times them."""

    name: ClassVar[str] = "offchip_traffic"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("offchip_traffic_weight",)

    def __init__(self, charge: bool = True) -> None:
        self.charge = charge

    def build(self, ctx: FormulationContext) -> None:
        space = ctx.space
        if self.charge and not space.shared_bandwidth and (bandwidth := space.offchip_bandwidth()):
            ctx.quantities.add("offchip_traffic_weight", space.iterations / bandwidth)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        q, space, z_stop = ctx.quantities, ctx.space, ctx.vars.z_stop
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
