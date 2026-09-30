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

# (coefficient, expression, upper bound of the expression): one stream's or node's bits on a port
Terms = list[tuple[float, Any, float]]


def _port_name(key: PortKey) -> str:
    return "_".join(str(part) for part in key)


class MemoryPorts:
    """The interval bound: a port's bits per iteration fit in its rate times the initiation interval. With
    ``burst``, the burst bound: each slot's bits on a port fit in its rate times that slot's latency."""

    name: ClassVar[str] = "memory_ports"

    def __init__(self, burst: bool = True) -> None:
        self.burst = burst

    def declare(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        for port in alloc.accelerator.ports:
            q.add("port_rate", port.bits_per_cycle, index=port.key)
        per_iteration: dict[PortKey, Terms] = defaultdict(list)
        per_slot: dict[tuple[PortKey, int], Terms] = defaultdict(list)
        for stream in dma_streams(alloc):
            for side in stream.sides:
                term = (stream.bits_per_cycle * side.share / side.efficiency, stream.gated, stream.latency_ub)
                per_iteration[side.port.key].append(term)
                per_slot[(side.port.key, stream.slot)].append(term)
        for traffic in node_traffic(alloc):
            per_iteration[traffic.port.key].append((traffic.bits, 1, 1))
            per_slot[(traffic.port.key, traffic.slot)].append((traffic.bits, 1, 1))

        for key, terms in per_iteration.items():
            q.add("port_demand", self._sum(alloc, terms), index=key)
        for key, terms in per_slot.items():
            q.add("port_demand_slot", self._sum(alloc, terms), index=key)
        for key, iteration_terms in per_iteration.items():
            rate = q.get("port_rate", key).expr
            demands = [iteration_terms, *(terms for (k, _), terms in per_slot.items() if k == key)]
            bound = max(sum(c * ub for c, _, ub in terms) for terms in demands) / rate
            cycles = q.get("port_demand", key).expr / rate
            # One cycle of margin against float rounding between this bound and the solved demand.
            q.add(SLOT_PRESSURE, cycles, index=("memory_ports", key), upper_bound=ceil(bound) + 1)

    def constrain(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        if "port_demand" not in q:
            return
        if alloc.constraint_selection.transfer_contention:
            interval = q.get("iteration").expr - q.get("overlap").expr
            for key, demand in q.indexed("port_demand").items():
                rate = q.get("port_rate", key).expr
                alloc.model.add_constr(rate * interval >= demand.expr, name=f"port_interval_{_port_name(key)}")
        if self.burst:
            for (key, slot), demand in q.indexed("port_demand_slot").items():
                alloc.model.add_constr(
                    q.get("port_rate", key).expr * q.get("slot_latency", slot).expr >= demand.expr,
                    name=f"port_burst_{_port_name(key)}_{slot}",
                )

    @staticmethod
    def _sum(alloc: TransferAndTensorAllocator, terms: Terms) -> Any:
        return alloc.model.quicksum(coefficient * value for coefficient, value, _ in terms)._raw
