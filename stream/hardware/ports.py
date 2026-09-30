"""The memory ports each core's backend declares, keyed by the physical memory they belong to."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from stream.cost_model.bandwidth import BandwidthModel

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core

PortKey = tuple[int, str, str]
PortRef = tuple[int, str, str]
Service = tuple[str, str]

OUTPUT = "output"
READ = "read"
WRITE = "write"
READ_BY_DATAPATH = "read_by_datapath"
WRITE_BY_DATAPATH = "write_by_datapath"


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
        return (self.share_group, self.memory, self.name)


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
        """Ports of every core; a core with a measured bandwidth has none, as that already covers its traffic."""
        found: dict[PortKey, PortSpec] = {}
        core_ids: dict[PortKey, set[int]] = defaultdict(set)
        serves: dict[PortKey, set[Service]] = defaultdict(set)
        key_of: dict[PortRef, PortKey] = {}
        for core in accelerator.cores.node_list:
            if core.id in accelerator.bandwidth:
                continue
            for spec in core.memory_ports():
                found.setdefault(spec.key, spec)
                core_ids[spec.key].add(core.id)
                serves[spec.key].update(spec.serves)
                key_of[(core.id, spec.memory, spec.name)] = spec.key
        overrides = _overrides_by_key(accelerator.port_bandwidth, key_of)
        ports = {
            key: Port(
                key=key,
                core_ids=tuple(sorted(core_ids[key])),
                memory=spec.memory,
                name=spec.name,
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
        """The port ``operand``'s top level uses for ``direction`` on ``core``, else any top-level port of
        ``core`` that serves ``direction``."""
        ports = self.ports_of(core)
        own = next((p for p in ports if (operand, direction) in p.serves), None)
        return own or next((p for p in ports if any(d == direction for _, d in p.serves)), None)


def _overrides_by_key(
    port_bandwidth: dict[PortRef, BandwidthModel], key_of: dict[PortRef, PortKey]
) -> dict[PortKey, BandwidthModel]:
    overrides: dict[PortKey, BandwidthModel] = {}
    for ref, model in port_bandwidth.items():
        if ref not in key_of:
            core_id, memory, port = ref
            raise ValueError(
                f"`bandwidth.{core_id}.{memory}.{port}`: core {core_id} declares no port `{port}` on a top-level "
                f"memory `{memory}`, or its backend models no memory ports."
            )
        if overrides.setdefault(key_of[ref], model) != model:
            raise ValueError(f"bandwidth entries disagree on shared port {key_of[ref]}.")
    return overrides
