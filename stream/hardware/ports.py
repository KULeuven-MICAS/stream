"""Memory ports: what each port serves, how fast it is and which cores reach it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from stream.cost_model.bandwidth import BandwidthModel

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator, MemoryGraph
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.memory import Memory
    from stream.hardware.architecture.noc.communication_link import CommunicationLink


Service = tuple[str, str]
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


BY_DATAPATH = {READ: READ_BY_DATAPATH, WRITE: WRITE_BY_DATAPATH}
"""For a transfer direction (read or write), the matching datapath direction."""


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
    """A port that a core's backend declares on its top-level memory instance ``memory``. ``serves`` holds
    (operand role, direction) pairs, ``bits_per_cycle`` is the peak rate and the energies are in pJ per bit."""

    memory: str
    name: str
    bits_per_cycle: float
    serves: frozenset[Service]
    read_energy_per_bit: float = 0.0
    write_energy_per_bit: float = 0.0


@dataclass(frozen=True)
class MemoryDma:
    """A memory's DMA engine. Its channels are the memory's ports that have ``channels``. ``buffer_descriptors`` is
    how many descriptors it has and ``iterations`` how many steps one descriptor can repeat."""

    buffer_descriptors: int
    iterations: int


@dataclass(frozen=True)
class Port:
    """One port of a memory. ``core_ids`` are the cores that reach it (empty for a port declared on the memory
    itself). ``serves`` holds (operand role, direction) pairs and the energies are in pJ per bit. ``channels`` is the
    number of DMA channels, each carrying one fifo at a time; None if the port is not a DMA."""

    key: PortKey
    core_ids: tuple[int, ...]
    bandwidth: BandwidthModel = field(hash=False)
    serves: frozenset[Service]
    read_energy_per_bit: float = 0.0
    write_energy_per_bit: float = 0.0
    channels: int | None = None

    def energy_per_bit(self, direction: str) -> float:
        return self.read_energy_per_bit if direction == READ else self.write_energy_per_bit

    @property
    def bits_per_cycle(self) -> float:
        """Peak rate of all reads and writes through the port together."""
        return self.bandwidth.ceiling


class PortRegistry:
    """Every memory port of the accelerator, and which of them each core reaches."""

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
        """The ports of every memory, with the measured bandwidths of ``accelerator.port_bandwidth`` applied. A memory
        with a measured bandwidth of its own has no ports here, since that measurement covers its traffic."""
        ports: dict[PortKey, Port] = {}
        for memory in accelerator.memories.values():
            if memory not in accelerator.bandwidth:
                ports |= {port.key: port for port in memory.ports.values()}
        unknown = sorted(set(accelerator.port_bandwidth) - set(ports))
        if unknown:
            raise ValueError(
                f"`bandwidth` names ports {['.'.join(key) for key in unknown]} no memory declares, or of a memory with "
                "a measured bandwidth."
            )
        for key, model in accelerator.port_bandwidth.items():
            ports[key] = replace(ports[key], bandwidth=model)
        return cls(ports)

    def check_connections(self, network: MemoryGraph, measured: Collection[Memory]) -> None:
        """Reject a connection that names, of a memory without a ``measured`` bandwidth, no port serving reads or none
        serving writes, since no port would then bound a transfer over it; or several ports of one memory serving the
        same operand and direction (read or write), since it is then unclear which port a transfer uses."""
        for link in dict.fromkeys(data["cl"] for _, _, data in network.edges(data=True)):
            for memory, names in link.ports.items():
                through = [p for p in self._ports_of_memory.get(memory.name, ()) if p.key.port in names]
                unserved = [d for d in (READ, WRITE) if not candidate_ports(through, d, ANY_OPERAND)]
                if unserved and memory not in measured:
                    raise ValueError(
                        f"{link} names no port of memory {memory.name} serving {' or '.join(unserved)}: name the ports "
                        "a transfer passes each way"
                    )
                services = {service for p in through for service in p.serves if service[1] in (READ, WRITE)}
                for operand, direction in sorted(services):
                    own = candidate_ports(through, direction, operand)
                    if len(own) > 1:
                        raise ValueError(
                            f"{link} names ports {sorted(p.key.port for p in own)} of memory {memory.name}, which all "
                            f"serve {operand} {direction}: name the one a transfer passes"
                        )

    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports.values())

    def ports_of(self, core: Core) -> tuple[Port, ...]:
        """The top-level ports ``core`` reaches; none for a core without a port model."""
        return self._ports_of.get(core.id, ())

    def port_for(self, core: Core, direction: str, operand: str, memory: Memory | None = None) -> Port | None:
        """The port of ``core`` that ``operand`` uses for ``direction``, limited to the ports on ``memory`` when given.
        Picks the first of :func:`candidate_ports`; None if there is none."""
        ports = self.ports_of(core)
        if memory is not None:
            ports = tuple(p for p in ports if p.key.memory == memory.name)
        serving = candidate_ports(ports, direction, operand)
        return serving[0] if serving else None

    def memory_ports(
        self, memory: Memory, direction: str, operand: str, links: Iterable[CommunicationLink]
    ) -> dict[Port, tuple[CommunicationLink, ...]]:
        """The ports of ``memory`` that a transfer of ``operand`` in ``direction`` passes over ``links``, each with the
        links that use it. Each link uses at most one port: of the ports it names on ``memory``, the one serving the
        operand."""
        by_port: dict[Port, tuple[CommunicationLink, ...]] = {}
        for link in links:
            names = set(link.ports.get(memory, ()))
            through = tuple(port for port in self._ports_of_memory.get(memory.name, ()) if port.key.port in names)
            port = _serving(through, direction, operand, f"memory {memory.name} over {link}")
            if port is not None:
                by_port[port] = (*by_port.get(port, ()), link)
        return by_port


def candidate_ports(ports: tuple[Port, ...], direction: str, operand: str) -> list[Port]:
    """The ports ``operand`` may use in ``direction``, by preference: those serving that operand (or every operand),
    else those serving the direction at all. For a read or write that no port serves, fall back to the ports the
    datapath uses in that direction, as a core's memory hierarchy may give an operand no port of its own."""
    own = [p for p in ports if (operand, direction) in p.serves or (ANY_OPERAND, direction) in p.serves]
    serving = own or [p for p in ports if any(d == direction for _, d in p.serves)]
    if not serving and direction in BY_DATAPATH:
        serving = [p for p in ports if any(d == BY_DATAPATH[direction] for _, d in p.serves)]
    return serving


def _serving(ports: tuple[Port, ...], direction: str, operand: str, where: str) -> Port | None:
    """The one port of :func:`candidate_ports`; None if there is none. Several candidates raise, as it is then unclear
    which port the transfer uses (:meth:`PortRegistry.check_connections` catches most of these at parse time)."""
    serving = candidate_ports(ports, direction, operand)
    if len(serving) > 1:
        names = sorted(p.key.port for p in serving)
        raise ValueError(f"{where}: ports {names} all serve {operand} {direction}; name the one a transfer passes")
    return serving[0] if serving else None


def build_memory_ports(
    memory: str, views: tuple[tuple[Core, str], ...], entries: dict[str, dict[str, Any]], complete: bool = True
) -> dict[str, Port]:
    """The ports of ``memory`` from its validated ``ports`` entries, by name. An entry with ``from`` takes the port of
    that name from the listed cores' backends, which must all declare it the same. Any other entry declares the port
    directly. If ``complete``, every port the viewing cores declare must be used by exactly one entry."""
    specs = {
        core.id: {spec.name: spec for spec in core.port_specs() if spec.memory == instance} for core, instance in views
    }
    ports = {name: _port(memory, name, entry, specs) for name, entry in entries.items()}
    if complete:
        _check_covered(memory, entries, specs)
    return ports


def _port(memory: str, name: str, entry: dict[str, Any], specs: dict[int, dict[str, PortSpec]]) -> Port:
    key = PortKey(memory, name)
    if "from" not in entry:
        return Port(
            key=key,
            core_ids=(),
            bandwidth=BandwidthModel.flat(entry["bandwidth"]),
            serves=frozenset(
                (role, direction) for role, directions in entry["serves"].items() for direction in directions
            ),
            read_energy_per_bit=entry["read_energy"],
            write_energy_per_bit=entry["write_energy"],
            channels=entry.get("channels"),
        )
    source, core_ids = entry["from"], tuple(entry["cores"])
    declared = [_view_spec(memory, core_id, source, specs) for core_id in core_ids]
    if len({(spec.bits_per_cycle, spec.read_energy_per_bit, spec.write_energy_per_bit) for spec in declared}) != 1:
        raise ValueError(
            f"memory {memory} port {name}: cores {list(core_ids)} declare {source} differently; one port's width and "
            "energies per bit must agree"
        )
    return Port(
        key=key,
        core_ids=tuple(sorted(core_ids)),
        bandwidth=BandwidthModel.flat(declared[0].bits_per_cycle),
        serves=frozenset(service for spec in declared for service in spec.serves),
        read_energy_per_bit=declared[0].read_energy_per_bit,
        write_energy_per_bit=declared[0].write_energy_per_bit,
    )


def _view_spec(memory: str, core_id: int, source: str, specs: dict[int, dict[str, PortSpec]]) -> PortSpec:
    """The port ``source`` that core ``core_id`` declares on its view of ``memory``."""
    if core_id not in specs:
        raise ValueError(f"memory {memory}: a port names core {core_id}, which is no view of it")
    if source not in specs[core_id]:
        raise ValueError(
            f"memory {memory}: core {core_id} declares no port {source} on its view of it; it declares "
            f"{sorted(specs[core_id])}"
        )
    return specs[core_id][source]


def _check_covered(memory: str, entries: dict[str, dict[str, Any]], specs: dict[int, dict[str, PortSpec]]) -> None:
    """Check that every port the viewing cores declare is used by exactly one entry."""
    taken: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name, entry in entries.items():
        for core_id in entry.get("cores", ()):
            taken[(core_id, entry["from"])].append(name)
    for core_id, declared in specs.items():
        for spec_name in declared:
            names = taken.get((core_id, spec_name), [])
            if len(names) != 1:
                raise ValueError(
                    f"memory {memory}: core {core_id}'s port {spec_name} is in ports {names}; each port a view's "
                    "core declares belongs to exactly one of its memory's ports"
                )
