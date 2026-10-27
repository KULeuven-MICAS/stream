"""The ports each memory declares, keyed by the memory: ports its cores' backends declare on their views of it, or
its own."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from stream.cost_model.bandwidth import BandwidthModel

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.memory import Memory
    from stream.hardware.architecture.noc.communication_link import CommunicationLink

logger = logging.getLogger(__name__)

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


BY_DATAPATH = {READ: READ_BY_DATAPATH, WRITE: WRITE_BY_DATAPATH}
"""The datapath's direction through a memory for each of a transfer's."""


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
    """A memory's port no core declares, as an off-chip memory's is; ``serves`` holds (operand role, direction).
    ``channels`` are the DMA channels the port is, as an AIE2 shim tile's DMA reaching DRAM, each carrying one fifo at
    a time; None for a port that is no DMA, which bounds only its bits per cycle."""

    bits_per_cycle: float
    serves: frozenset[Service]
    read_energy_per_bit: float
    write_energy_per_bit: float
    channels: int | None = None


@dataclass(frozen=True)
class MemoryDma:
    """The DMA a memory's own ports are the channels of, as the AIE2 shim tile's DMA that reaches DRAM: the buffer
    descriptors it has for the runtime sequence and the steps one descriptor iterates."""

    buffer_descriptors: int
    iterations: int


@dataclass(frozen=True)
class Port:
    """One port of a memory, keyed by (memory, port name), with the cores that reach it.
    ``serves`` holds (operand role, direction) and the energies are in pJ per bit; ``channels`` are the DMA channels
    the port is, None where it is no DMA."""

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
        bandwidth has no ports, as that already covers its traffic, so it need not declare its cores' either."""
        ports: dict[PortKey, Port] = {}
        for memory in accelerator.memories.values():
            if memory not in accelerator.bandwidth:
                ports |= {port.key: port for port in declared_ports(memory)}
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

    def check_connections(self, network: Any) -> None:
        """Warn, once at parse time, of each connection end naming several ports that all serve one operand in one of
        a transfer's directions (a read or a write; the datapath's are each core's own): a transfer passing the end
        that way leaves which port its bits take undeclared, and :meth:`memory_port` raises on it when one does. A
        description may still mean it, as a connection through two cores' own ports that only one core's transfers
        take, so it is no error here."""
        for link in dict.fromkeys(data["cl"] for _, _, data in network.edges(data=True)):
            for memory, names in link.ports.items():
                through = [p for p in self._ports_of_memory.get(memory.name, ()) if p.key.port in names]
                services = {service for p in through for service in p.serves if service[1] in (READ, WRITE)}
                for operand, direction in sorted(services):
                    own = [
                        p for p in through if (operand, direction) in p.serves or (ANY_OPERAND, direction) in p.serves
                    ]
                    if len(own) > 1:
                        logger.warning(
                            "%s names ports %s of memory %s, which all serve %s %s: a transfer passing them that way "
                            "fails to say which it takes; name the one it passes",
                            link,
                            [p.key.port for p in own],
                            memory.name,
                            operand,
                            direction,
                        )

    def __iter__(self) -> Iterator[Port]:
        return iter(self._ports.values())

    def ports_of(self, core: Core) -> tuple[Port, ...]:
        """The top-level ports ``core`` reaches; none for a core without a port model."""
        return self._ports_of.get(core.id, ())

    def port_for(self, core: Core, direction: str, operand: str, memory: Memory | None = None) -> Port | None:
        """The port ``operand``'s top level uses for ``direction`` on ``core``, as its datapath reaches it: of the
        core's ports on ``memory``, the one holding the operand, where given (None where that memory has no modelled
        port, as a measured one), else of all its ports; the first serving that operand (or every operand) in that
        direction, else the first serving the direction at all, else, for a transfer's read or write, the first port
        the datapath reads or writes the memory through (a read port is one whichever side of it the hierarchy
        allocates), as a ZigZag hierarchy allocates an input no port out of its top level but its bits still pass
        one; the datapath takes no transfer's port, as a tile DMA, so its unserved access passes none. A core's own
        ports leave nothing else to decide by, so several candidates are no error here; which port a transfer passes
        a memory through is :meth:`memory_port`'s, where a connection names it."""
        ports = self.ports_of(core)
        if memory is not None:
            ports = tuple(p for p in ports if p.key.memory == memory.name)
        serving = _candidates(ports, direction, operand)
        return serving[0] if serving else None

    def memory_ports(
        self, memory: Memory, direction: str, operand: str, links: Iterable[CommunicationLink]
    ) -> dict[Port, tuple[CommunicationLink, ...]]:
        """The ports of ``memory`` a transfer moving ``operand`` passes in ``direction`` over ``links``, each with the
        links passing it: per link, of the ports it enters and leaves ``memory`` through, the one serving the operand
        (a route over several links, one per column's shim, passes one port per link; a link naming two that serve it
        alike passes the first by name, see :func:`_serving`). Empty where no link reaches the memory or none of its
        ports serves the direction."""
        by_port: dict[Port, tuple[CommunicationLink, ...]] = {}
        for link in links:
            names = set(link.ports.get(memory, ()))  # a link not joining the memory passes none of its ports
            through = tuple(port for port in self._ports_of_memory.get(memory.name, ()) if port.key.port in names)
            if (port := _serving(through, direction, operand, f"memory {memory.name} over {link}")) is not None:
                by_port[port] = (*by_port.get(port, ()), link)
        return by_port

    def memory_port(
        self, memory: Memory, direction: str, operand: str, links: Iterable[CommunicationLink]
    ) -> Port | None:
        """The one port of ``memory`` a transfer moving ``operand`` passes in ``direction`` over ``links``
        (:meth:`memory_ports`); None where no link reaches it, and an error where the links pass several ports, which
        the caller must then weigh apart."""
        links = tuple(links)
        ports = self.memory_ports(memory, direction, operand, links)
        if len(ports) > 1:
            names = sorted(port.key.port for port in ports)
            raise ValueError(f"memory {memory.name} over {sorted(map(str, links))}: the links pass ports {names}")
        return next(iter(ports), None)


def _candidates(ports: tuple[Port, ...], direction: str, operand: str) -> list[Port]:
    """Of ``ports``, those the bits of ``operand`` may pass in ``direction``: the ones serving that operand (or every
    operand) in it, else the ones serving the direction at all, else, for a transfer's read or write, the ones the
    datapath reads or writes the memory through (a read port is one whichever side of it the hierarchy allocates);
    the datapath takes no transfer's port, as a tile DMA, so its unserved access has none."""
    own = [p for p in ports if (operand, direction) in p.serves or (ANY_OPERAND, direction) in p.serves]
    serving = own or [p for p in ports if any(d == direction for _, d in p.serves)]
    if not serving and direction in BY_DATAPATH:
        serving = [p for p in ports if any(d == BY_DATAPATH[direction] for _, d in p.serves)]
    return serving


def _serving(ports: tuple[Port, ...], direction: str, operand: str, where: str) -> Port | None:
    """The port of ``where`` a transfer passes the bits of ``operand`` in ``direction``, of :func:`_candidates`; None
    where none serves it. Several candidates, as views naming their ports apart declare on one connection, leave which
    the bits pass undeclared: the first by name is taken, the same one every time, as :meth:`PortRegistry
    .check_connections` warned at parse time; a connection naming only the one it passes settles it."""
    serving = _candidates(ports, direction, operand)
    if len(serving) > 1:
        serving.sort(key=lambda port: port.key.port)
        logger.debug(
            "%s: ports %s all serve %s %s; taking %s",
            where,
            [p.key.port for p in serving],
            operand,
            direction,
            serving[0].key.port,
        )
    return serving[0] if serving else None


ViewSpecs = dict[int, dict[str, PortSpec]]
"""Per core viewing a memory, the ports its backend declares on its view of it, by name."""


def declared_ports(memory: Memory) -> tuple[Port, ...]:
    """The ports ``memory`` declares, at their declared widths, after checking every port a view's core declares is
    in exactly one of them. Each view's backend is asked for its ports once."""
    specs = _view_specs(memory)
    _check_covered(memory, specs)
    return tuple(_built(memory, name, declared, specs) for name, declared in memory.ports.items())


def _view_specs(memory: Memory) -> ViewSpecs:
    """The ports each view's core declares on its view of ``memory``, by core id and port name."""
    return {
        core.id: {spec.name: spec for spec in core.port_specs() if spec.memory == instance}
        for core, instance in memory.views
    }


def _built(memory: Memory, name: str, declared: ViewPort | OwnPort, specs: ViewSpecs) -> Port:
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
            channels=declared.channels,
        )
    view_specs = [_view_spec(memory, core_id, declared.source, specs) for core_id in declared.core_ids]
    rates = {(spec.bits_per_cycle, spec.read_energy_per_bit, spec.write_energy_per_bit) for spec in view_specs}
    if len(rates) != 1:
        raise ValueError(
            f"memory {memory.name} port {name}: cores {list(declared.core_ids)} declare {declared.source} differently; "
            "one port's width and energies per bit must agree"
        )
    return Port(
        key=key,
        core_ids=tuple(sorted(declared.core_ids)),
        bandwidth=BandwidthModel.flat(view_specs[0].bits_per_cycle),
        serves=frozenset(service for spec in view_specs for service in spec.serves),
        read_energy_per_bit=view_specs[0].read_energy_per_bit,
        write_energy_per_bit=view_specs[0].write_energy_per_bit,
    )


def _view_spec(memory: Memory, core_id: int, source: str, specs: ViewSpecs) -> PortSpec:
    """The port ``source`` core ``core_id``'s backend declares on its view of ``memory``."""
    if core_id not in specs:
        raise ValueError(f"memory {memory.name}: a port names core {core_id}, which is no view of it")
    if source not in specs[core_id]:
        raise ValueError(
            f"memory {memory.name}: core {core_id} declares no port {source} on {memory.instance_of(core_id)}; it "
            f"declares {sorted(specs[core_id])}"
        )
    return specs[core_id][source]


def _check_covered(memory: Memory, specs: ViewSpecs) -> None:
    """Every port a view's core declares is in exactly one of the memory's ports."""
    taken: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name, declared in memory.ports.items():
        if isinstance(declared, ViewPort):
            for core_id in declared.core_ids:
                taken[(core_id, declared.source)].append(name)
    for core_id, declared_specs in specs.items():
        for spec_name in declared_specs:
            if len(names := taken.get((core_id, spec_name), [])) != 1:
                raise ValueError(
                    f"memory {memory.name}: core {core_id}'s port {spec_name} is in ports {names}; each port a view's "
                    "core declares belongs to exactly one of its memory's ports"
                )
