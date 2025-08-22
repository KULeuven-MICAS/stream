from enum import Enum

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor


class TransferType(Enum):
    """Enumeration for different types of data transfer operations."""

    UNICAST = "unicast"
    BROADCAST = "broadcast"
    JOIN = "join"


class SteadyStateTransfer(SteadyStateNode):
    """A node representing a data transfer operation in the graph."""

    def __init__(  # noqa: PLR0913
        self,
        id: int,
        node_name: str,
        transfer_type: TransferType,
        srcs: tuple[SteadyStateTensor, ...],
        dsts: tuple[SteadyStateTensor, ...],
        size: int,  # in bits
        tensor: SteadyStateTensor,
        possible_resource_allocation: tuple[tuple[CommunicationLink, ...], ...],
        steady_state_iteration_space: SteadyStateIterationSpace,
    ):
        super().__init__(
            id=id,
            node_name=node_name,
            type="transfer",
            possible_resource_allocation=possible_resource_allocation,
            steady_state_iteration_space=steady_state_iteration_space,
        )
        self.srcs = srcs
        self.dsts = dsts
        self.tensor = tensor
        self.size = size
        self.transfer_type = transfer_type
        if self.possible_resource_allocation:
            assert len(self.possible_resource_allocation) > 0, "Possible resource allocation must not be empty."
            self.possible_resource_allocation: tuple[tuple[CommunicationLink, ...], ...] = possible_resource_allocation
            self.chosen_resource_allocation: tuple[CommunicationLink, ...] | None = (
                possible_resource_allocation[0] if len(possible_resource_allocation) == 1 else None
            )
        self.chosen_memory_core: Core | None = None

    def set_possible_resource_allocation(self, allocation: tuple[tuple[CommunicationLink, ...], ...]) -> None:
        assert len(allocation) > 0, "Allocation must not be empty."
        self.possible_resource_allocation = allocation
        self.chosen_resource_allocation = None if len(allocation) > 1 else allocation[0]

    def __str__(self):
        return f"Transfer({self.srcs} -> {self.dsts})"

    def __repr__(self):
        return str(self)

    @property
    def plot_name(self):
        return f"Transfer({self.srcs})"
