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
MemoryView = tuple[int, str]
"""A core's view of a memory: (core id, the name of its top-level memory instance)."""

OUTPUT = "output"
READ = "read"
WRITE = "write"
READ_BY_DATAPATH = "read_by_datapath"
WRITE_BY_DATAPATH = "write_by_datapath"


class PortKey(NamedTuple):
    """A physical port: the memory it belongs to (`Memory.name`) and the port's name, the same on every view of the
    memory that reaches it."""

    memory: str
    port: str


ANY_OPERAND = "any"
"""Operand role of a port that serves every operand, such as a tile DMA."""


def input_role(k: int) -> str:
    """Operand role of the k-th input of a node, counted from 1."""
    return f"input{k}"


@dataclass(frozen=True)
class MemoryBanks:
    """A memory split into ``count`` banks of ``width``-byte words, each holding ``stripe`` consecutive words before
    the next bank does: byte address ``a`` sits in bank ``a // width // stripe % count``, so a stripe of 1 interleaves
    words and a bank's depth gives each bank one contiguous range. Declared per core (``banks:``) and reported with each
    layout; the allocation does not price bank conflicts."""

    count: int
    width: int
    stripe: int = 1


@dataclass(frozen=True)
class PortSpec:
    """One port of a core's top-level memory instance ``memory`` as its backend declares it. ``serves`` holds
    (operand role, direction), ``bits_per_cycle`` is the declared peak rate and the energies are in pJ per bit."""

    memory: str
    name: str
    bits_per_cycle: float
    serves: frozenset[Service]
    read_energy_per_bit: float = 0.0
    write_energy_per_bit: float = 0.0


@dataclass(frozen=True)
class Port:
    """One port of a memory, keyed by (memory, port name), with the cores that reach it.
    ``serves`` holds (operand role, direction) and the energies are in pJ per bit."""

    key: PortKey
    core_ids: tuple[int, ...]
    bandwidth: BandwidthModel = field(hash=False)
    serves: frozenset[Service]
    read_energy_per_bit: float = 0.0
    write_energy_per_bit: float = 0.0

    def energy_per_bit(self, direction: str) -> float:
        return self.read_energy_per_bit if direction == READ else self.write_energy_per_bit

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
        """The ports of every memory: those each core's backend declares on the instance its view of the memory
        binds, the same port when two views name it alike. A core's other instances are its own, and a memory with a
        measured bandwidth has no ports, as that already covers its traffic."""
        found: dict[PortKey, PortSpec] = {}
        core_ids: dict[PortKey, set[int]] = defaultdict(set)
        serves: dict[PortKey, set[Service]] = defaultdict(set)
        for core in accelerator.core_list:
            memory = accelerator.memory_of(core)
            if memory in accelerator.bandwidth:
                continue
            instance = memory.instance(core)
            for spec in (spec for spec in core.port_specs() if spec.memory == instance):
                key = PortKey(memory.name, spec.name)
                if found.setdefault(key, spec).bits_per_cycle != spec.bits_per_cycle:
                    raise ValueError(
                        f"the views of memory {memory.name} declare port {spec.name} at different rates: "
                        f"{found[key].bits_per_cycle} and {spec.bits_per_cycle} bits per cycle"
                    )
                core_ids[key].add(core.id)
                serves[key].update(spec.serves)
        if unknown := sorted(set(accelerator.port_bandwidth) - set(found)):
            raise ValueError(
                f"`bandwidth` names ports {['.'.join(key) for key in unknown]} that no core models: the memory has no "
                f"such port, or its cores' backends model no memory ports."
            )
        ports = {
            key: Port(
                key=key,
                core_ids=tuple(sorted(core_ids[key])),
                bandwidth=accelerator.port_bandwidth.get(key, BandwidthModel.flat(spec.bits_per_cycle)),
                serves=frozenset(serves[key]),
                read_energy_per_bit=spec.read_energy_per_bit,
                write_energy_per_bit=spec.write_energy_per_bit,
            )
            for key, spec in sorted(found.items())
        }
        return cls(ports)

    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports.values())

    def ports_of(self, core: Core) -> tuple[Port, ...]:
        """The top-level ports ``core`` reaches; none for a core without a port model."""
        return self._ports_of.get(core.id, ())

    def port_for(self, core: Core, direction: str, operand: str) -> Port | None:
        """The port ``operand``'s top level uses for ``direction`` on ``core``.
        When no port serves that pair, any port of ``core`` serving ``direction``, as the bits still pass one."""
        ports = self.ports_of(core)
        own = next((p for p in ports if (operand, direction) in p.serves), None)
        return own or next((p for p in ports if any(d == direction for _, d in p.serves)), None)
