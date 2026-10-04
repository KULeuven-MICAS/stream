"""Memory-port bandwidth: what each top-level port moves must fit in the time the schedule gives it."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.ports import PortKey
from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE
from stream.opt.allocation.constraint_optimization.families.traffic import DmaStream, dma_streams, node_traffic

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
    """With ``interval``, the interval bound: a port's bits per iteration fit in its rate times the initiation
    interval, for every port. With ``burst``, the burst bound: each slot's bits on a port fit in its rate times
    that slot's latency. With neither, the model is unchanged and the family only reports. Rates in bits/cycle."""

    name: ClassVar[str] = "memory_ports"
    declare_requires: ClassVar[tuple[str, ...]] = ("transfer_latency",)
    declares: ClassVar[tuple[str, ...]] = ("port_rate", "port_demand", "port_demand_slot", SLOT_PRESSURE)
    requires: ClassVar[tuple[str, ...]] = ("iteration", "overlap")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, interval: bool = True, burst: bool = True) -> None:
        self.interval = interval
        self.burst = burst

    def declare(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        for port in alloc.accelerator.ports:
            q.add("port_rate", port.bits_per_cycle, index=port.key)
        per_iteration: dict[PortKey, Terms] = defaultdict(list)
        per_slot: dict[tuple[PortKey, int], Terms] = defaultdict(list)
        for stream in dma_streams(alloc):
            for side in stream.sides:
                term = (stream.bits_per_cycle * side.share / side.efficiency, stream.active_cycles, stream.latency_ub)
                per_iteration[side.port.key].append(term)
                per_slot[(side.port.key, stream.slot)].append(term)
        for traffic in node_traffic(alloc):
            per_iteration[traffic.port.key].append((traffic.bits, 1, 1))
            per_slot[(traffic.port.key, traffic.slot)].append((traffic.bits, 1, 1))

        for key, terms in per_iteration.items():
            q.add("port_demand", self._sum(alloc, terms), index=key)
        for key, terms in per_slot.items():
            q.add("port_demand_slot", self._sum(alloc, terms), index=key)
        if not (self.interval or self.burst):
            return
        for key, iteration_terms in per_iteration.items():
            rate = q.get("port_rate", key).expr
            demands = [iteration_terms, *(terms for (k, _), terms in per_slot.items() if k == key)]
            bound = max(sum(c * ub for c, _, ub in terms) for terms in demands) / rate
            cycles = q.get("port_demand", key).expr / rate
            # One cycle of margin against float rounding between this bound and the solved demand.
            q.add(SLOT_PRESSURE, cycles, index=("memory_ports", key), upper_bound=ceil(bound) + 1)

    def build(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        if "port_demand" not in q:
            return
        if self.interval:
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

    def report(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> dict[str, Any]:
        """Per resource in ZigZag's port-activity terms, busiest first: ``real_cycle`` it needs for one iteration's
        bits, ``allowed_cycle`` the initiation interval, and ``stall_or_slack`` their difference. Resources are
        memory ports, measured shared bandwidth and links; the last two are read from the solved transfers."""
        interval = alloc.model.value(q.get("iteration").expr) - alloc.model.value(q.get("overlap").expr)
        rows = [*self._port_rows(alloc, q, interval), *_shared_rows(alloc, q, interval), *_link_rows(alloc, interval)]
        rows.sort(key=lambda row: -(row["utilization"] or 0.0))
        return {"memory_ports": rows}

    @staticmethod
    def _port_rows(alloc: TransferAndTensorAllocator, q: QuantityRegistry, interval: float) -> list[dict[str, Any]]:
        if "port_demand" not in q:
            return []
        value = alloc.model.value
        cores = {port.key: port.core_ids for port in alloc.accelerator.ports}
        rows = []
        for key, demand in q.indexed("port_demand").items():
            rate = q.get("port_rate", key).expr
            bursts = {
                slot: value(d.expr) / (rate * alloc.slot_latency[slot].X)
                for (k, slot), d in q.indexed("port_demand_slot").items()
                if k == key and alloc.slot_latency[slot].X > 0
            }
            busiest = max(bursts, key=bursts.__getitem__, default=None)
            row = _activity_row(
                alloc, "memory_port", f"{key.memory}.{key.port}", cores[key], rate, value(demand.expr), interval
            )
            rows.append(
                row | {"burst_utilization": bursts.get(busiest) if busiest is not None else None, "burst_slot": busiest}
            )
        return rows

    @staticmethod
    def _sum(alloc: TransferAndTensorAllocator, terms: Terms) -> Any:
        return alloc.model.quicksum(coefficient * value for coefficient, value, _ in terms)._raw


def _activity_row(
    alloc: TransferAndTensorAllocator,
    kind: str,
    name: str,
    core_ids: tuple[int, ...],
    rate: float,
    bits: float,
    interval: float,
) -> dict[str, Any]:
    core_types = [alloc.accelerator.get_core(core_id).core_type for core_id in core_ids]
    real_cycle = bits / rate
    return {
        "kind": kind,
        "resource": name,
        "core_ids": list(core_ids),
        "core_types": core_types,
        "bw_bits_per_cycle": rate,
        "bits_per_iteration": bits,
        "req_bw_aver": bits / interval if interval > 0 else None,
        "real_cycle": real_cycle,
        "allowed_cycle": interval,
        "stall_or_slack": real_cycle - interval,
        "utilization": real_cycle / interval if interval > 0 else None,
        "burst_utilization": None,
        "burst_slot": None,
    }


def _stream_bits(alloc: TransferAndTensorAllocator) -> list[tuple[DmaStream, float]]:
    return [(stream, alloc.model.value(stream.bits_per_cycle * stream.active_cycles)) for stream in dma_streams(alloc)]


def _shared_rows(alloc: TransferAndTensorAllocator, q: QuantityRegistry, interval: float) -> list[dict[str, Any]]:
    """A core with a measured bandwidth: its solved busy time, which counts the access pattern's slow down."""
    rows = []
    streams = _stream_bits(alloc)
    for core_id, model in alloc.shared_bandwidth.items():
        bits = sum(b for s, b in streams if any(c.id == core_id for c in (*s.choice.sources, *s.choice.targets)))
        row = _activity_row(alloc, "shared_bandwidth", "measured", (core_id,), model.ceiling, bits, interval)
        if ("shared_busy" in q) and core_id in q.indexed("shared_busy"):
            busy = alloc.model.value(q.get("shared_busy", core_id).expr)
            row |= {
                "real_cycle": busy,
                "stall_or_slack": busy - interval,
                "utilization": busy / interval if interval > 0 else None,
            }
        rows.append(row)
    return rows


def _link_rows(alloc: TransferAndTensorAllocator, interval: float) -> list[dict[str, Any]]:
    bits: dict[Any, float] = defaultdict(float)
    for stream, stream_bits in _stream_bits(alloc):
        for link in stream.choice.links_used:
            bits[link] += stream_bits
    return [
        _activity_row(alloc, "link", _link_name(link), _link_cores(link), link.bandwidth, b, interval)
        for link, b in bits.items()
        if b > 0
    ]


def _link_name(link: Any) -> str:
    return "->".join(str(getattr(end, "id", end)) for end in (link.sender, link.receiver))


def _link_cores(link: Any) -> tuple[int, ...]:
    return tuple(end.id for end in (link.sender, link.receiver) if hasattr(end, "id"))
