from enum import Enum

from stream.hardware.architecture.core import Core
from stream.workload.steady_state_iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state_node import SteadyStateNode
from stream.workload.steady_state_tensor import SteadyStateTensor


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
        src: object,
        dst: object,
        tensor: SteadyStateTensor,
        possible_resource_allocation: list[Core | None],
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

    def __str__(self):
        return f"Transfer({self.src} -> {self.dst})"

    def __repr__(self):
        return str(self)

    @property
    def plot_name(self):
        return f"Transfer({self.src})"
