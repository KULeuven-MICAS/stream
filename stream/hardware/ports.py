"""The ports each memory declares, keyed by the memory: ports its cores' backends declare on their views of it, or
its own."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple

from stream.cost_model.bandwidth import BandwidthModel

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.memory import Memory
    from stream.hardware.architecture.noc.communication_link import CommunicationLink

Service = tuple[str, str]
MemoryView = tuple[int, str]
"""A core's view of a memory: (core id, the name of its top-level memory instance)."""

OUTPUT = "output"
READ = "read"
WRITE = "write"
READ_BY_DATAPATH = "read_by_datapath"
WRITE_BY_DATAPATH = "write_by_datapath"


class PortKey(NamedTuple):
    """A physical port: the memory it belongs to (`Memory.name`) and the name that memory declares it by."""

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
class ViewPort:
    """A memory's port that is the port ``source`` the backends of ``core_ids`` declare on their views of it, one
    physical port they all reach."""

    source: str
    core_ids: tuple[int, ...]


@dataclass(frozen=True)
class OwnPort:
    """A memory's port no core declares, as an off-chip memory's is; ``serves`` holds (operand role, direction)."""

    bits_per_cycle: float
    serves: frozenset[Service]
    read_energy_per_bit: float
    write_energy_per_bit: float


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
        self._ports_of_memory: dict[str, tuple[Port, ...]] = defaultdict(tuple)
        for port in ports.values():
            self._ports_of_memory[port.key.memory] += (port,)
            for core_id in port.core_ids:
                self._ports_of[core_id] += (port,)

    @classmethod
    def from_accelerator(cls, accelerator: Accelerator) -> PortRegistry:
        """The ports every memory declares: one a port its views' cores declare, reached by the cores it names, or
        one of its own. Every port a view's core declares is in exactly one of its memory's. A memory with a measured
        bandwidth has no ports, as that already covers its traffic."""
        ports: dict[PortKey, Port] = {}
        for memory in accelerator.memories.values():
            built = declared_ports(memory)
            if memory not in accelerator.bandwidth:
                ports |= {port.key: port for port in built}
        if unknown := sorted(set(accelerator.port_bandwidth) - set(ports)):
            raise ValueError(
                f"`bandwidth` names ports {['.'.join(key) for key in unknown]} no memory declares, or of a memory with "
                "a measured bandwidth."
            )
        return cls(
            {
                key: replace(port, bandwidth=model) if (model := accelerator.port_bandwidth.get(key)) else port
                for key, port in ports.items()
            }
        )

    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports.values())

    def ports_of(self, core: Core) -> tuple[Port, ...]:
        """The top-level ports ``core`` reaches; none for a core without a port model."""
        return self._ports_of.get(core.id, ())

    def port_for(self, core: Core, direction: str, operand: str) -> Port | None:
        """The port ``operand``'s top level uses for ``direction`` on ``core``, as its datapath reaches it."""
        return _serving(self.ports_of(core), direction, operand, f"core {core.id}")

    def memory_port(
        self, memory: Memory, direction: str, operand: str, links: Iterable[CommunicationLink]
    ) -> Port | None:
        """The port of ``memory`` a transfer moving ``operand`` passes in ``direction`` over ``links``: of the ports
        those links enter and leave ``memory`` through, the one serving it. None where no link reaches it."""
        links = tuple(links)
        names = {name for link in links for name in link.ports[memory]}
        through = tuple(port for port in self._ports_of_memory.get(memory.name, ()) if port.key.port in names)
        return _serving(through, direction, operand, f"memory {memory.name} over {sorted(map(str, links))}")


def _serving(ports: tuple[Port, ...], direction: str, operand: str, where: str) -> Port | None:
    """The port of ``where`` the bits of ``operand`` pass in ``direction``: the one serving that operand (or every
    operand) there, else the one port serving the direction at all, as a ZigZag hierarchy allocates an input no port
    out of its top level; None where none serves the direction. Several candidates, as views naming their ports apart
    declare, leave which the bits pass undeclared, and raise."""
    own = [p for p in ports if (operand, direction) in p.serves or (ANY_OPERAND, direction) in p.serves]
    serving = own or [p for p in ports if any(d == direction for _, d in p.serves)]
    if len(serving) > 1:
        names = [p.key.port for p in serving]
        raise ValueError(f"{where}: ports {names} all serve {operand} {direction}; name the one a transfer passes")
    return serving[0] if serving else None


def declared_ports(memory: Memory) -> tuple[Port, ...]:
    """The ports ``memory`` declares, at their declared widths, after checking every port a view's core declares is
    in exactly one of them."""
    _check_covered(memory)
    return tuple(_built(memory, name, declared) for name, declared in memory.ports.items())


def _built(memory: Memory, name: str, declared: ViewPort | OwnPort) -> Port:
    """The port ``name`` of ``memory`` as declared: its own, or the port its cores' backends declare on their views of
    it, which they must declare alike."""
    key = PortKey(memory.name, name)
    if isinstance(declared, OwnPort):
        return Port(
            key=key,
            core_ids=(),
            bandwidth=BandwidthModel.flat(declared.bits_per_cycle),
            serves=declared.serves,
            read_energy_per_bit=declared.read_energy_per_bit,
            write_energy_per_bit=declared.write_energy_per_bit,
        )
    specs = [_view_spec(memory, core_id, declared.source) for core_id in declared.core_ids]
    rates = {(spec.bits_per_cycle, spec.read_energy_per_bit, spec.write_energy_per_bit) for spec in specs}
    if len(rates) != 1:
        raise ValueError(
            f"memory {memory.name} port {name}: cores {list(declared.core_ids)} declare {declared.source} differently; "
            "one port's width and energies per bit must agree"
        )
    return Port(
        key=key,
        core_ids=tuple(sorted(declared.core_ids)),
        bandwidth=BandwidthModel.flat(specs[0].bits_per_cycle),
        serves=frozenset(service for spec in specs for service in spec.serves),
        read_energy_per_bit=specs[0].read_energy_per_bit,
        write_energy_per_bit=specs[0].write_energy_per_bit,
    )


def _view_spec(memory: Memory, core_id: int, source: str) -> PortSpec:
    """The port ``source`` core ``core_id``'s backend declares on its view of ``memory``."""
    views = {core.id: (core, instance) for core, instance in memory.views}
    if core_id not in views:
        raise ValueError(f"memory {memory.name}: a port names core {core_id}, which is no view of it")
    core, instance = views[core_id]
    declared = {spec.name: spec for spec in core.port_specs() if spec.memory == instance}
    if source not in declared:
        raise ValueError(
            f"memory {memory.name}: core {core_id} declares no port {source} on {instance}; it declares "
            f"{sorted(declared)}"
        )
    return declared[source]


def _check_covered(memory: Memory) -> None:
    """Every port a view's core declares is in exactly one of the memory's ports."""
    taken: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name, declared in memory.ports.items():
        if isinstance(declared, ViewPort):
            for core_id in declared.core_ids:
                taken[(core_id, declared.source)].append(name)
    for core, instance in memory.views:
        for spec in (spec for spec in core.port_specs() if spec.memory == instance):
            if len(names := taken.get((core.id, spec.name), [])) != 1:
                raise ValueError(
                    f"memory {memory.name}: core {core.id}'s port {spec.name} is in ports {names}; each port a view's "
                    "core declares belongs to exactly one of its memory's ports"
                )
