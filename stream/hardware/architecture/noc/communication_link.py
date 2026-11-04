from __future__ import annotations

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory


class CommunicationLink:
    """A fixed-bandwidth connection between memories: either a one-way link from ``sender`` to ``receiver``, or a bus
    shared by its ``members`` in every direction. Build one with :meth:`link` or :meth:`bus`. ``parallel`` numbers the
    connections declared between the same memories, each a separate resource. ``ports`` gives, per memory, the names
    of the ports the connection uses on it."""

    def __init__(
        self,
        sender: Memory | None,
        receiver: Memory | None,
        members: tuple[Memory, ...],
        bandwidth: int | float,
        unit_energy_cost: float,
        parallel: int,
        ports: dict[Memory, tuple[str, ...]],
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.members = members
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.parallel = parallel
        self.ports = ports
        self._key = (sender, receiver, members, bandwidth, unit_energy_cost, parallel)

    @classmethod
    def link(  # noqa: PLR0913
        cls,
        sender: Memory,
        receiver: Memory,
        bandwidth: int | float,
        unit_energy_cost: float,
        parallel: int,
        ports: dict[Memory, tuple[str, ...]],
    ) -> CommunicationLink:
        return cls(sender, receiver, (), bandwidth, unit_energy_cost, parallel, ports)

    @classmethod
    def bus(
        cls,
        members: tuple[Memory, ...],
        bandwidth: int | float,
        unit_energy_cost: float,
        parallel: int,
        ports: dict[Memory, tuple[str, ...]],
    ) -> CommunicationLink:
        return cls(None, None, members, bandwidth, unit_energy_cost, parallel, ports)

    @property
    def memories(self) -> tuple[Memory, ...]:
        """The memories the connection joins: a bus's members, else the link's two ends."""
        return self.members or (self.sender, self.receiver)

    @property
    def cores(self) -> tuple[Core, ...]:
        """The cores of the memories the connection joins."""
        return tuple(sorted({core for memory in self.memories for core in memory.cores}, key=lambda core: core.id))

    def __str__(self) -> str:
        nth = f"#{self.parallel}" if self.parallel else ""
        if self.members:
            return f"Bus{nth}({','.join(memory.name for memory in self.members)}, bw={self.bandwidth})"
        return f"CL{nth}({self.sender.name}, {self.receiver.name}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CommunicationLink) and self._key == other._key

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CommunicationLink):
            return NotImplemented
        return (self.memories, self.parallel) < (other.memories, other.parallel)
