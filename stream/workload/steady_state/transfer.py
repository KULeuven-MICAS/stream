from enum import Enum

from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor


class TransferType(Enum):
    """Enumeration for different types of data transfer operations."""

    UNICAST = "unicast"
    BROADCAST = "broadcast"


class SteadyStateTransfer(SteadyStateNode):
    """A node representing a data transfer operation in the graph."""

    def __init__(
        self,
        id: int,
        node_name: str,
        transfer_type: TransferType,
        src: SteadyStateTensor,
        dst: SteadyStateTensor | SteadyStateComputation,
        tensor: SteadyStateTensor,
        possible_resource_allocation: tuple[tuple[CommunicationLink]],
        steady_state_iteration_space: SteadyStateIterationSpace,
    ):
        super().__init__(
            id=id,
            node_name=node_name,
            type="transfer",
            possible_resource_allocation=possible_resource_allocation,
            steady_state_iteration_space=steady_state_iteration_space,
        )
        self.src = src
        self.dst = dst
        self.tensor = tensor
        self.size = tensor.size
        self.transfer_type = transfer_type
        self.possible_resource_allocation: tuple[tuple[CommunicationLink]]
        self.chosen_resource_allocation: tuple[CommunicationLink] | None = (
            None if len(possible_resource_allocation) > 1 else possible_resource_allocation[0]
        )

    def __str__(self):
        return f"Transfer({self.src} -> {self.dst})"

    def __repr__(self):
        return str(self)

    @property
    def plot_name(self):
        return f"Transfer({self.src})"
