"""The ports of each ZigZag core's top-level memories, keyed by the physical memory they belong to."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.memory_port import DataDirection, MemoryPort

from stream.cost_model.bandwidth import BandwidthModel

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core

PortKey = tuple[int, str, str]
PortRef = tuple[int, str, str]
Service = tuple[str, int, str]


@dataclass(frozen=True)
class Port:
    """One port of a top-level memory, keyed by (shared memory group id, memory instance, port name).
    Energies are in pJ per bit; ``serves`` holds (memory operand, level index, ``DataDirection`` name)."""

    key: PortKey
    core_ids: tuple[int, ...]
    memory: str
    name: str
    bandwidth: BandwidthModel = field(hash=False)
    read_energy_per_bit: float
    write_energy_per_bit: float
    serves: frozenset[Service]

    @property
    def bits_per_cycle(self) -> float:
        """Peak rate of all reads and writes through the port together."""
        return self.bandwidth.ceiling


def has_port_model(core: Core, accelerator: Accelerator) -> bool:
    """Only ZigZag cores describe ports; a measured core bandwidth already covers the core's traffic."""
    return core.namespace == "zigzag" and core.id not in accelerator.bandwidth


def top_levels(core: Core) -> dict[str, tuple[MemoryLevel, int]]:
    """Memory operand -> its top memory level on ``core`` and that level's index."""
    hierarchy = core.to_zigzag_core().memory_hierarchy
    levels_of = {str(op): hierarchy.get_memory_levels(op) for op in hierarchy.get_operands()}
    return {op: (levels[-1], len(levels) - 1) for op, levels in levels_of.items()}


class PortRegistry:
    """Every top-level port of the accelerator, and which of them a core reaches."""

    def __init__(self, ports: dict[PortKey, Port], top_level_index: dict[int, dict[str, int]]):
        self._ports = ports
        self._top_level_index = top_level_index
        self._ports_of = {
            core_id: tuple(port for port in ports.values() if core_id in port.core_ids) for core_id in top_level_index
        }

    @classmethod
    def from_accelerator(cls, accelerator: Accelerator) -> PortRegistry:
        tops = {core.id: top_levels(core) for core in accelerator.cores.node_list if has_port_model(core, accelerator)}
        found: dict[PortKey, tuple[MemoryLevel, MemoryPort]] = {}
        core_ids: dict[PortKey, set[int]] = defaultdict(set)
        serves: dict[PortKey, set[Service]] = defaultdict(set)
        key_of: dict[PortRef, PortKey] = {}
        for core_id, levels in tops.items():
            for level, _ in levels.values():
                for port in level.ports:
                    key = (level.memory_instance.shared_memory_group_id, level.name, port.name)
                    found.setdefault(key, (level, port))
                    core_ids[key].add(core_id)
                    serves[key].update((str(op), lv, direction.name) for op, lv, direction in port.served_op_lv_dir)
                    key_of[(core_id, level.name, port.name)] = key
        overrides = _overrides_by_key(accelerator.port_bandwidth, key_of)
        ports = {
            key: Port(
                key=key,
                core_ids=tuple(sorted(core_ids[key])),
                memory=level.name,
                name=port.name,
                bandwidth=overrides.get(key, BandwidthModel.flat(float(port.bw_max))),
                read_energy_per_bit=level.memory_instance.r_cost / port.bw_max,
                write_energy_per_bit=level.memory_instance.w_cost / port.bw_max,
                serves=frozenset(serves[key]),
            )
            for key, (level, port) in sorted(found.items())
        }
        index = {core_id: {op: lv for op, (_, lv) in levels.items()} for core_id, levels in tops.items()}
        return cls(ports, index)

    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports.values())

    def __len__(self) -> int:
        return len(self._ports)

    def ports_of(self, core: Core) -> tuple[Port, ...]:
        """The top-level ports ``core`` reaches; none for a core without a port model."""
        return self._ports_of.get(core.id, ())

    def port_for(self, core: Core, direction: DataDirection, memory_operand: str) -> Port | None:
        """The port ``memory_operand``'s top level uses for ``direction`` on ``core``, else any top-level
        port of ``core`` that serves ``direction``."""
        index = self._top_level_index.get(core.id)
        if index is None:
            return None
        own = (str(memory_operand), index.get(str(memory_operand), -1), direction.name)
        any_top = {(op, lv, direction.name) for op, lv in index.items()}
        ports = self.ports_of(core)
        return next((p for p in ports if own in p.serves), None) or next((p for p in ports if p.serves & any_top), None)


def _overrides_by_key(
    port_bandwidth: dict[PortRef, BandwidthModel], key_of: dict[PortRef, PortKey]
) -> dict[PortKey, BandwidthModel]:
    overrides: dict[PortKey, BandwidthModel] = {}
    for ref, model in port_bandwidth.items():
        if ref not in key_of:
            raise ValueError(f"bandwidth entry {ref} names no top-level port of a core with a port model.")
        if overrides.setdefault(key_of[ref], model) != model:
            raise ValueError(f"bandwidth entries disagree on shared port {key_of[ref]}.")
    return overrides
