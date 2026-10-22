from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.memory import Memory


class CommunicationLink:
    """A fixed-bandwidth connection between memories: a link carrying data from ``sender`` to ``receiver`` (a declared
    link is one each way), or a bus its ``members`` share in every direction. ``parallel`` tells apart the connections
    declared between the same memories, each its own resource. Build one with :meth:`link` or :meth:`bus`."""

    def __init__(
        self,
        sender: Memory | None,
        receiver: Memory | None,
        members: tuple[Memory, ...],
        bandwidth: int | float,
        unit_energy_cost: float,
        parallel: int,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.members = members
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.parallel = parallel
        self._key = (sender, receiver, members, bandwidth, unit_energy_cost, parallel)

    @classmethod
    def link(
        cls, sender: Memory, receiver: Memory, bandwidth: int | float, unit_energy_cost: float, parallel: int
    ) -> CommunicationLink:
        return cls(sender, receiver, (), bandwidth, unit_energy_cost, parallel)

    @classmethod
    def bus(
        cls, members: tuple[Memory, ...], bandwidth: int | float, unit_energy_cost: float, parallel: int
    ) -> CommunicationLink:
        return cls(None, None, members, bandwidth, unit_energy_cost, parallel)

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
        return self._order() < other._order()

    def _order(self) -> tuple[str, ...]:
        return (*(memory.name for memory in self.memories), str(self.parallel))
