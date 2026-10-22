from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.memory import Memory


def get_bidirectional_edges(
    memory_a: "Memory",
    memory_b: "Memory",
    bandwidth: float,
    unit_energy_cost: float,
    link_type: Literal["bus"] | Literal["link"],
    bus_instance: "CommunicationLink | None" = None,
    parallel: int = 0,
) -> list[tuple["Memory", "Memory", dict[str, "CommunicationLink"]]]:
    """Create a list with two edges: from A to B and B to A, the ``parallel``-th link between them."""
    bus = bus_instance or CommunicationLink("Any", "Any", bandwidth, unit_energy_cost, bidirectional=True)
    link_a_to_b = CommunicationLink(memory_a, memory_b, bandwidth, unit_energy_cost, parallel=parallel)
    link_b_to_a = CommunicationLink(memory_b, memory_a, bandwidth, unit_energy_cost, parallel=parallel)

    return [
        #  A -> B
        (
            memory_a,
            memory_b,
            {"cl": bus if link_type == "bus" else link_a_to_b},
        ),
        # B -> A
        (
            memory_b,
            memory_a,
            {"cl": bus if link_type == "bus" else link_b_to_a},
        ),
    ]


class CommunicationLink:
    """A fixed-bandwidth connection moving data between memories: a link from ``sender`` to ``receiver``, or a bus
    between ``members`` (its ends then "Any"). ``parallel`` tells apart the connections between the same memories,
    each its own resource."""

    def __init__(
        self,
        sender: "Memory | str",
        receiver: "Memory | str",
        bandwidth: int | float,
        unit_energy_cost: float,
        bidirectional: bool = False,
        members: tuple["Memory", ...] = (),
        parallel: int = 0,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.bidirectional = bidirectional  # TODO this property is not in use?
        self.members = members
        self.parallel = parallel
        self._hash = hash(
            (
                self.sender,
                self.receiver,
                self.bandwidth,
                self.unit_energy_cost,
                self.bidirectional,
                self.members,
                self.parallel,
            )
        )

    @property
    def memories(self) -> tuple["Memory", ...]:
        """The memories the link connects: a bus's members, else its two ends."""
        return self.members or tuple(end for end in (self.sender, self.receiver) if not isinstance(end, str))

    @property
    def cores(self) -> tuple["Core", ...]:
        """The cores of the memories the link connects."""
        return tuple(sorted({core for memory in self.memories for core in memory.cores}, key=lambda core: core.id))

    def __str__(self) -> str:
        nth = f"#{self.parallel}" if self.parallel else ""
        if self.members:
            return f"Bus{nth}({','.join(memory.name for memory in self.members)}, bw={self.bandwidth})"
        return f"CL{nth}({self.sender.name}, {self.receiver.name}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CommunicationLink) and (
            self.sender,
            self.receiver,
            self.bandwidth,
            self.members,
            self.parallel,
        ) == (other.sender, other.receiver, other.bandwidth, other.members, other.parallel)

    def _order(self) -> tuple[str, ...]:
        return (str(self.sender), str(self.receiver), *(memory.name for memory in self.members), str(self.parallel))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CommunicationLink):
            return NotImplemented
        return self._order() < other._order()
