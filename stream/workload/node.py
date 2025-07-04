from abc import ABCMeta

from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.workload.layer_node_abc import LayerNodeABC


class Node(LayerNodeABC, metaclass=ABCMeta):
    """Abstract base class that represents a piece of an algorithmic workload.
    Example: ComputationNode, etc.
    """

    offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving]

    def __init__(  # noqa: PLR0913
        self,
        node_id: int,
        node_name: str,
        type: str,
        onchip_energy: float,
        offchip_energy: float,
        runtime: int,
        possible_core_allocation: list[int],
        chosen_core_allocation: int | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the Node metaclass

        Args:
            type: The type of Node.
            energy: The energy consumption of this Node.
            runtime: The runtime of this Node.
            possible_core_allocation: The core id on which this Node can be mapped.
            inputs: The names of the input tensors of this node
            outputs: The names of the output tensors of this node.
            chosen_core_allocation: The final core allocation of this node
            input_names: Names of the ONNX input node
        """
        if input_names is None:
            input_names = []
        super().__init__(node_id, node_name)

        self.type = type.lower()
        self.onchip_energy = onchip_energy
        self.offchip_energy = offchip_energy
        self.runtime = runtime
        self.possible_core_allocation = possible_core_allocation
        self.chosen_core_allocation = chosen_core_allocation
        self.input_names = input_names
        self.start = -1
        self.end = -1
        # number of data (in bits) only this node produces (not produced by any other node)
        self.data_produced_unique = 0

    def get_total_energy(self) -> float:
        """Get the total energy of running this node, including off-chip energy."""
        return self.onchip_energy + self.offchip_energy

    def get_onchip_energy(self):
        """Get the on-chip energy of running this node."""
        return self.onchip_energy

    def get_offchip_energy(self):
        """Get the off-chip energy of running this node."""
        return self.offchip_energy

    def get_runtime(self):
        """Get the runtime of running this node."""
        return self.runtime

    def get_start(self):
        """Get the start time in cycles of this node."""
        return self.start

    def get_end(self):
        """Get the end time in cycles of this node."""
        return self.end

    def set_onchip_energy(self, energy: float):
        """Set the on-chip energy of running this node.

        Args:
            energy (float): energy consumption of this node
        """
        self.onchip_energy = energy

    def set_offchip_energy(self, energy: float):
        """Set the off-chip energy of running this node.

        Args:
            energy (float): energy consumption of this node
        """
        self.offchip_energy = energy

    def set_runtime(self, runtime: int):
        """Set the runtime of running this node.

        Args:
            runtime: runtime in cycles
        """
        self.runtime = runtime

    def set_start(self, start: int):
        """Set the start time in cycles of this node.

        Args:
            start: start time in cycles
        """
        self.start = start

    def set_end(self, end: int):
        """Set the end time in cycles of this node.

        Args:
            end: end time in cycles
        """
        self.end = end

    def set_core_allocation(self, core_allocation: int):
        self.core_allocation = [core_allocation]

    def set_chosen_core_allocation(self, core_allocation: int | None):
        self.chosen_core_allocation = core_allocation

    def has_end(self) -> bool:
        """Check if this node has already been assigned an end time.

        Returns:
            bool: True if this node has been assigned an end time
        """
        return self.end is not None

    def set_offchip_bandwidth(self, offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving]):
        self.offchip_bandwidth_per_op = offchip_bandwidth_per_op

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
