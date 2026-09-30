"""The memory ports each core's backend declares, keyed by the physical memory they belong to."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from stream.cost_model.bandwidth import BandwidthModel

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core

Service = tuple[str, str]
MemoryRef = tuple[int, str]

OUTPUT = "output"
READ = "read"
WRITE = "write"
READ_BY_DATAPATH = "read_by_datapath"
WRITE_BY_DATAPATH = "write_by_datapath"


class PortKey(NamedTuple):
    """A physical port: the share group and name of the memory that owns it, and the port name."""

    share_group: int
    memory: str
    port: str


class PortRef(NamedTuple):
    """A port as one core names it, as in a ``bandwidth:`` port key."""

    core_id: int
    memory: str
    port: str


def input_role(k: int) -> str:
    """Operand role of the k-th input of a node, counted from 1."""
    return f"input{k}"


@dataclass(frozen=True)
class PortSpec:
    """One port of a core's top-level memory as its backend declares it. ``serves`` holds (operand role,
    direction); energies are in pJ per bit and ``bits_per_cycle`` is the declared peak rate."""

    share_group: int
    memory: str
    name: str
    bits_per_cycle: float
    read_energy_per_bit: float
    write_energy_per_bit: float
    serves: frozenset[Service]

    @property
    def key(self) -> PortKey:
        return PortKey(self.share_group, self.memory, self.name)


@dataclass(frozen=True)
class Port:
    """One port of a top-level memory, keyed by (share group, memory, port name), with the cores that reach it.
    Energies are in pJ per bit; ``serves`` holds (operand role, direction)."""

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


class PortRegistry:
    """Every top-level port of the accelerator, and which of them a core reaches."""

    def __init__(self, ports: dict[PortKey, Port]):
        self._ports = ports
        self._ports_of: dict[int, tuple[Port, ...]] = defaultdict(tuple)
        for port in ports.values():
            for core_id in port.core_ids:
                self._ports_of[core_id] += (port,)

    @classmethod
    def from_accelerator(cls, accelerator: Accelerator) -> PortRegistry:
        """Ports of every core, one per port of each physical memory; a core with a measured bandwidth has none,
        as that already covers its traffic. An aliased memory takes the ports of its group's first memory."""
        specs = {core.id: core.memory_ports() for core in accelerator.cores.node_list}
        share_group = {(core_id, spec.memory): spec.share_group for core_id in specs for spec in specs[core_id]}
        found: dict[PortKey, PortSpec] = {}
        core_ids: dict[PortKey, set[int]] = defaultdict(set)
        serves: dict[PortKey, set[Service]] = defaultdict(set)
        key_of: dict[PortRef, PortKey] = {}
        for core_id, core_specs in specs.items():
            if core_id in accelerator.bandwidth:
                continue
            for spec in core_specs:
                owner = accelerator.memory_aliases.get((core_id, spec.memory), (core_id, spec.memory))
                key = PortKey(share_group.get(owner, spec.share_group), owner[1], spec.name)
                if key not in found or owner == (core_id, spec.memory):
                    found[key] = spec
                core_ids[key].add(core_id)
                serves[key].update(spec.serves)
                key_of[PortRef(core_id, spec.memory, spec.name)] = key
        overrides = _overrides_by_key(accelerator.port_bandwidth, key_of)
        ports = {
            key: Port(
                key=key,
                core_ids=tuple(sorted(core_ids[key])),
                memory=key.memory,
                name=key.port,
                bandwidth=overrides.get(key, BandwidthModel.flat(spec.bits_per_cycle)),
                read_energy_per_bit=spec.read_energy_per_bit,
                write_energy_per_bit=spec.write_energy_per_bit,
                serves=frozenset(serves[key]),
            )
            for key, spec in sorted(found.items())
        }
        return cls(ports)

    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports.values())

    def __len__(self) -> int:
        return len(self._ports)

    def ports_of(self, core: Core) -> tuple[Port, ...]:
        """The top-level ports ``core`` reaches; none for a core without a port model."""
        return self._ports_of.get(core.id, ())

    def port_for(self, core: Core, direction: str, operand: str) -> Port | None:
        """The port ``operand``'s top level uses for ``direction`` on ``core``.
        When no port serves that pair, any port of ``core`` serving ``direction``, as the bits still pass one."""
        ports = self.ports_of(core)
        own = next((p for p in ports if (operand, direction) in p.serves), None)
        return own or next((p for p in ports if any(d == direction for _, d in p.serves)), None)


def _overrides_by_key(
    port_bandwidth: dict[PortRef, BandwidthModel], key_of: dict[PortRef, PortKey]
) -> dict[PortKey, BandwidthModel]:
    overrides: dict[PortKey, BandwidthModel] = {}
    for ref, model in port_bandwidth.items():
        if ref not in key_of:
            raise ValueError(
                f"`bandwidth.{ref.core_id}.{ref.memory}.{ref.port}`: core {ref.core_id} declares no port "
                f"`{ref.port}` on a top-level memory `{ref.memory}`, or its backend models no memory ports."
            )
        if overrides.setdefault(key_of[ref], model) != model:
            raise ValueError(f"bandwidth entries disagree on shared port {key_of[ref]}.")
    return overrides
