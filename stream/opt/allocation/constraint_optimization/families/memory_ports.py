"""Memory-port bandwidth: what each top-level port moves must fit in the time the schedule gives it."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.ports import PortKey
from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE
from stream.opt.allocation.constraint_optimization.families.traffic import dma_streams, node_traffic

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

Terms = list[tuple[float, Any]]


def _port_name(key: PortKey) -> str:
    return "_".join(str(part) for part in key)


class MemoryPorts:
    """P2 bounds each port's bits per iteration by its rate times the initiation interval; with ``burst``,
    P1 also bounds each slot's bits by its rate times that slot's latency. Adds no variables."""

    name: ClassVar[str] = "memory_ports"

    def __init__(self, burst: bool = True) -> None:
        self.burst = burst
        self.rate: dict[PortKey, float] = {}

    def declare(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        self.rate = {port.key: port.bits_per_cycle for port in alloc.accelerator.ports}
        per_iteration: dict[PortKey, Terms] = defaultdict(list)
        per_slot: dict[tuple[PortKey, int], Terms] = defaultdict(list)
        worst: dict[Any, float] = {}
        for stream in dma_streams(alloc):
            worst[id(stream.gated)] = stream.latency_ub
            for side in stream.sides:
                term = (stream.bits_per_cycle * side.share / side.efficiency, stream.gated)
                per_iteration[side.port.key].append(term)
                per_slot[(side.port.key, stream.slot)].append(term)
        for traffic in node_traffic(alloc):
            per_iteration[traffic.port.key].append((traffic.bits, 1))
            per_slot[(traffic.port.key, traffic.slot)].append((traffic.bits, 1))

        for key, terms in per_iteration.items():
            q.add("port_demand", self._sum(alloc, terms), index=key)
        for key, terms in per_slot.items():
            q.add("port_demand_slot", self._sum(alloc, terms), index=key)
        for key, iteration_terms in per_iteration.items():
            demands = [iteration_terms, *(terms for (k, _), terms in per_slot.items() if k == key)]
            bound = max(self._upper(terms, worst) for terms in demands) / self.rate[key]
            cycles = q.get("port_demand", key).expr / self.rate[key]
            q.add(SLOT_PRESSURE, cycles, index=("memory_ports", key), upper_bound=ceil(bound) + 1)

    def constrain(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        if "port_demand" not in q:
            return
        if alloc.constraint_selection.transfer_contention:
            interval = q.get("iteration").expr - q.get("overlap").expr
            for key, demand in q.indexed("port_demand").items():
                alloc.model.add_constr(
                    self.rate[key] * interval >= demand.expr, name=f"port_interval_{_port_name(key)}"
                )
        if self.burst:
            for (key, slot), demand in q.indexed("port_demand_slot").items():
                alloc.model.add_constr(
                    self.rate[key] * q.get("slot_latency", slot).expr >= demand.expr,
                    name=f"port_burst_{_port_name(key)}_{slot}",
                )

    @staticmethod
    def _sum(alloc: TransferAndTensorAllocator, terms: Terms) -> Any:
        return alloc.model.quicksum(coefficient * value for coefficient, value in terms)._raw

    @staticmethod
    def _upper(terms: Terms, worst: dict[Any, float]) -> float:
        """Demand when every gated stream takes its longest latency."""
        return sum(c * (v if isinstance(v, int | float) else worst[id(v)]) for c, v in terms)
