from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core

ENABLE_BROADCASTING = False


def get_bidirectional_edges(
    core_a: "Core",
    core_b: "Core",
    bandwidth: float,
    unit_energy_cost: float,
    link_type: Literal["bus"] | Literal["link"],
    bus_instance: "CommunicationLink | None" = None,
) -> list[tuple["Core", "Core", dict[str, "CommunicationLink"]]]:
    """Create a list with two edges: from A to B and B to A."""
    bus = bus_instance or CommunicationLink("Any", "Any", bandwidth, unit_energy_cost, bidirectional=True)
    link_a_to_b = CommunicationLink(core_a, core_b, bandwidth, unit_energy_cost)
    link_b_to_a = CommunicationLink(core_b, core_a, bandwidth, unit_energy_cost)

    # if have_shared_memory(core_a, core_b):
    #     # No edge if the cores have a shared memory
    #     return []

    return [
        #  A -> B
        (
            core_a,
            core_b,
            {"cl": bus if link_type == "bus" else link_a_to_b},
        ),
        # B -> A
        (
            core_b,
            core_a,
            {"cl": bus if link_type == "bus" else link_b_to_a},
        ),
    ]


class CommunicationLink:
    """Represents a fixed-bandwidth communication link used to communicate between two cores."""

    def __init__(
        self,
        sender: "Core | Literal['Any']",
        receiver: "Core | Literal['Any']",
        bandwidth: int | float,
        unit_energy_cost: float,
        bidirectional: bool = False,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.bidirectional = bidirectional  # TODO this property is not in use?

    def __str__(self) -> str:
        return f"CL({self.sender}, {self.receiver}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.sender, self.receiver, self.bandwidth, self.unit_energy_cost, self.bidirectional))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CommunicationLink) and (self.sender, self.receiver, self.bandwidth) == (
            other.sender,
            other.receiver,
            other.bandwidth,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CommunicationLink):
            return NotImplemented
        sender_self_id = self.sender.id
        sender_other_id = other.sender.id
        if sender_self_id != sender_other_id:
            return sender_self_id < sender_other_id

        receiver_self_id = self.receiver.id
        receiver_other_id = other.receiver.id
        return receiver_self_id < receiver_other_id

    def get_name_for_schedule_plot(self) -> str:
        if self.bidirectional:
            return f"{self.sender} <-> {self.receiver}"
        else:
            return f"{self.sender} -> {self.receiver}"
