"""Memory-port bandwidth: what each top-level port moves must fit in the time the schedule gives it."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.ports import READ, PortKey
from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE
from stream.opt.allocation.constraint_optimization.families.latency import single_copy
from stream.opt.allocation.constraint_optimization.families.traffic import (
    DmaStream,
    dma_streams,
    link_cores,
    link_name,
    node_traffic,
    port_name,
)

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext

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
    declares: ClassVar[tuple[str, ...]] = ("port_rate", "port_bits", "port_demand", "port_demand_slot", SLOT_PRESSURE)
    requires: ClassVar[tuple[str, ...]] = ("iteration", "overlap")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, interval: bool = True, burst: bool = True) -> None:
        self.interval = interval
        self.burst = burst

    def declare(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        for port in ctx.space.accelerator.ports:
            q.add("port_rate", port.bits_per_cycle, index=port.key)
        per_iteration: dict[PortKey, Terms] = defaultdict(list)
        per_slot: dict[tuple[PortKey, int], Terms] = defaultdict(list)
        steady_iteration: dict[PortKey, Terms] = defaultdict(list)
        steady_slot: dict[tuple[PortKey, int], Terms] = defaultdict(list)
        for stream in dma_streams(ctx):
            steady = _steady_state_cycles(ctx, stream)
            for side in stream.sides:
                rate = stream.bits_per_cycle * side.share / side.efficiency
                term = (rate, stream.active_cycles, stream.latency_ub)
                per_iteration[side.port.key].append(term)
                per_slot[(side.port.key, stream.slot)].append(term)
                term = term if steady is None else (rate, steady, stream.latency_ub)
                steady_iteration[side.port.key].append(term)
                steady_slot[(side.port.key, stream.slot)].append(term)
        for traffic in node_traffic(ctx):
            term = (traffic.bits, 1, 1)
            per_iteration[traffic.port.key].append(term)
            per_slot[(traffic.port.key, traffic.slot)].append(term)
            steady = [] if ctx.space.runs_in_epilogue(traffic.node) else [term]
            steady_iteration[traffic.port.key] += steady
            steady_slot[(traffic.port.key, traffic.slot)] += steady
        for key, terms in per_iteration.items():
            q.add("port_bits", self._sum(ctx, terms), index=key)
        for key, terms in steady_iteration.items():
            q.add("port_demand", self._sum(ctx, terms), index=key)
        for key, terms in steady_slot.items():
            q.add("port_demand_slot", self._sum(ctx, terms), index=key)
        if not (self.interval or self.burst):
            return
        for key, iteration_terms in per_iteration.items():
            rate = q.get("port_rate", key).expr
            demands = [iteration_terms, *(terms for (k, _), terms in per_slot.items() if k == key)]
            bound = max(sum(c * ub for c, _, ub in terms) for terms in demands) / rate
            cycles = q.get("port_demand", key).expr / rate
            # One cycle of margin against float rounding between this bound and the solved demand.
            q.add(SLOT_PRESSURE, cycles, index=("memory_ports", key), upper_bound=ceil(bound) + 1)

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        if "port_demand" not in q:
            return
        if self.interval:
            interval = q.get("iteration").expr - q.get("overlap").expr
            for key, demand in q.indexed("port_demand").items():
                rate = q.get("port_rate", key).expr
                ctx.model.add_constr(rate * interval >= demand.expr, name=f"port_interval_{_port_name(key)}")
        if self.burst:
            for (key, slot), demand in q.indexed("port_demand_slot").items():
                ctx.model.add_constr(
                    q.get("port_rate", key).expr * q.get("slot_latency", slot).expr >= demand.expr,
                    name=f"port_burst_{_port_name(key)}_{slot}",
                )

    def report(self, ctx: FormulationContext) -> dict[str, Any]:
        """Per resource in ZigZag's port-activity terms, busiest first: ``real_cycle`` it needs for one iteration's
        bits, ``allowed_cycle`` the initiation interval, and ``stall_or_slack`` their difference. Resources are
        memory ports, measured shared bandwidth and links; the last two are read from the solved transfers."""
        q = ctx.quantities
        interval = ctx.model.value(q.get("iteration").expr) - ctx.model.value(q.get("overlap").expr)
        streams = _stream_bits(ctx)
        rows = [
            *self._port_rows(ctx, interval, streams),
            *_shared_rows(ctx, interval, streams),
            *_link_rows(ctx, interval, streams),
        ]
        rows.sort(key=lambda row: -(row["utilization"] or 0.0))
        return {"memory_ports": rows}

    @staticmethod
    def _port_rows(
        ctx: FormulationContext, interval: float, streams: list[tuple[DmaStream, tuple[float, float]]]
    ) -> list[dict[str, Any]]:
        q = ctx.quantities
        if "port_bits" not in q:
            return []
        value = ctx.model.value
        cores = {port.key: port.core_ids for port in ctx.space.accelerator.ports}
        once_per_run = _port_once(ctx, streams)
        rows = []
        for key, bits in q.indexed("port_bits").items():
            rate = q.get("port_rate", key).expr
            bursts = {
                slot: value(d.expr) / (rate * ctx.vars.slot_latency[slot].X)
                for (k, slot), d in q.indexed("port_demand_slot").items()
                if k == key and ctx.vars.slot_latency[slot].X > 0
            }
            busiest = max(bursts, key=bursts.__getitem__, default=None)
            once = once_per_run[key]
            steady = value(bits.expr) - once / ctx.space.iterations
            row = _activity_row(ctx, "memory_port", port_name(key), cores[key], rate, (steady, once), interval)
            rows.append(
                row | {"burst_utilization": bursts.get(busiest) if busiest is not None else None, "burst_slot": busiest}
            )
        return rows

    @staticmethod
    def _sum(ctx: FormulationContext, terms: Terms) -> Any:
        return ctx.model.quicksum(coefficient * value for coefficient, value, _ in terms)._raw


def _steady_state_cycles(ctx: FormulationContext, stream: DmaStream) -> Any:
    """The steady-state cycles per iteration of ``stream``, without a firing in the epilogue; None where it has none."""
    q = ctx.quantities
    if stream.copies is not None or "steady_state_latency" not in q:
        return None
    steady = q.indexed("steady_state_latency").get((stream.transfer, stream.choice))
    return None if steady is None else steady.expr


def _activity_row(
    ctx: FormulationContext,
    kind: str,
    name: str,
    core_ids: tuple[int, ...],
    rate: float,
    moved: tuple[float, float],
    interval: float,
) -> dict[str, Any]:
    """A resource's row from the bits it ``moved``: per iteration, and once in the run after the last iteration."""
    bits, once = moved
    core_types = [ctx.space.accelerator.get_core(core_id).core_type for core_id in core_ids]
    real_cycle = bits / rate
    return {
        "kind": kind,
        "resource": name,
        "core_ids": list(core_ids),
        "core_types": core_types,
        "bw_bits_per_cycle": rate,
        "bits_per_iteration": bits,
        "once_per_run_bits": once,
        "req_bw_aver": bits / interval if interval > 0 else None,
        "real_cycle": real_cycle,
        "allowed_cycle": interval,
        "stall_or_slack": real_cycle - interval,
        "utilization": real_cycle / interval if interval > 0 else None,
        "burst_utilization": None,
        "burst_slot": None,
    }


def _stream_bits(ctx: FormulationContext) -> list[tuple[DmaStream, tuple[float, float]]]:
    """Per stream, the bits it moves per iteration and those it moves once in the run, after the last iteration."""
    space = ctx.space
    found = []
    for stream in dma_streams(ctx):
        active = ctx.model.value(stream.active_cycles)
        tr, choice = stream.transfer, stream.choice
        held = 0.0
        if _steady_state_cycles(ctx, stream) is not None:
            copy = single_copy(ctx, tr, choice)
            stopped = sum(ctx.vars.z_stop[(copy, s)].X for s in space.once_per_run_stops(tr, copy))
            held = min(active, space.transfer_latency_for_path(tr, choice) / space.iterations * stopped)
        bits = stream.bits_per_cycle
        found.append((stream, (bits * (active - held), bits * held * space.iterations)))
    return found


def _port_once(ctx: FormulationContext, streams: list[tuple[DmaStream, tuple[float, float]]]) -> dict[PortKey, float]:
    """Per port, the bits it moves once in the run, after the last iteration."""
    once: dict[PortKey, float] = defaultdict(float)
    for stream, (_, stream_once) in streams:
        for side in stream.sides:
            once[side.port.key] += stream_once * side.share / side.efficiency
    for traffic in node_traffic(ctx):
        if ctx.space.runs_in_epilogue(traffic.node):
            once[traffic.port.key] += traffic.bits * ctx.space.iterations
    return once


def _shared_rows(
    ctx: FormulationContext, interval: float, streams: list[tuple[DmaStream, tuple[float, float]]]
) -> list[dict[str, Any]]:
    """A memory with a measured bandwidth: its solved busy time, which counts the access pattern's slow down."""
    q, space = ctx.quantities, ctx.space
    rows = []
    for memory, model in space.shared_bandwidth.items():
        shares = [
            (b, space.memory_share(s.transfer, s.choice, memory, direction == READ, s.copies))
            for s, b in streams
            if (direction := space.direction(memory, s.choice)) is not None
        ]
        bits = (sum(b[0] * share for b, share in shares), sum(b[1] * share for b, share in shares))
        cores = tuple(c.id for c in memory.cores)
        row = _activity_row(ctx, "shared_bandwidth", memory.name, cores, model.ceiling, bits, interval)
        if ("shared_busy" in q) and memory in q.indexed("shared_busy"):
            busy = ctx.model.value(q.get("shared_busy", memory).expr)
            row |= {
                "real_cycle": busy,
                "stall_or_slack": busy - interval,
                "utilization": busy / interval if interval > 0 else None,
            }
        rows.append(row)
    return rows


def _link_rows(
    ctx: FormulationContext, interval: float, streams: list[tuple[DmaStream, tuple[float, float]]]
) -> list[dict[str, Any]]:
    steady: dict[Any, float] = defaultdict(float)
    once: dict[Any, float] = defaultdict(float)
    for stream, stream_bits in streams:
        shares = dict(stream.link_shares) or dict.fromkeys(stream.choice.links_used, 1.0)
        for link in stream.choice.links_used:
            steady[link] += stream_bits[0] * shares.get(link, 0.0)
            once[link] += stream_bits[1] * shares.get(link, 0.0)
    return [
        _activity_row(ctx, "link", link_name(link), link_cores(link), link.bandwidth, (b, once[link]), interval)
        for link, b in steady.items()
        if b > 0 or once[link] > 0
    ]
