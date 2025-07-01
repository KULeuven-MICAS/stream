from abc import ABC
from typing import Optional

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.steady_state_iteration_space import SteadyStateIterationSpace


class SteadyStateNode(ABC):
    """Abstract base class for nodes in the steady state graph."""

    def __init__(
        self,
        id: int,
        node_name: str,
        type: str,
        steady_state_iteration_space: SteadyStateIterationSpace,
        possible_resource_allocation: None | list[Core] | tuple[tuple[CommunicationLink]],
        onchip_energy: Optional[float] = None,
        offchip_energy: Optional[float] = None,
        runtime: Optional[float] = None,
        chosen_resource_allocation: Optional[Core] = None,
        input_names: Optional[list[str]] = None,
    ):
        self.id = id
        self.node_name = node_name
        self.type = type.lower()
        self.steady_state_iteration_space = steady_state_iteration_space
        self._onchip_energy = onchip_energy
        self._offchip_energy = offchip_energy
        self._runtime = runtime
        self.possible_resource_allocation = possible_resource_allocation
        self._chosen_resource_allocation = chosen_resource_allocation
        self._input_names = input_names if input_names is not None else []

    @property
    def onchip_energy(self) -> Optional[float]:
        return self._onchip_energy

    @onchip_energy.setter
    def onchip_energy(self, value: Optional[float]) -> None:
        self._onchip_energy = value

    @property
    def offchip_energy(self) -> Optional[float]:
        return self._offchip_energy

    @offchip_energy.setter
    def offchip_energy(self, value: Optional[float]) -> None:
        self._offchip_energy = value

    @property
    def runtime(self) -> Optional[float]:
        return self._runtime

    @runtime.setter
    def runtime(self, value: Optional[float]) -> None:
        self._runtime = value

    @property
    def chosen_resource_allocation(self) -> Optional[Core]:
        return self._chosen_resource_allocation

    @chosen_resource_allocation.setter
    def chosen_resource_allocation(self, value: Optional[Core]) -> None:
        self._chosen_resource_allocation = value

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @input_names.setter
    def input_names(self, value: list[str]) -> None:
        self._input_names = value or []
