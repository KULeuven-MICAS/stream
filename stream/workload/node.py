from abc import ABCMeta

from zigzag.workload.LayerNodeABC import LayerNodeABC


class Node(LayerNodeABC, metaclass=ABCMeta):
    """Abstract base class that represents a piece of an algorithmic workload.
    Example: ComputationNode, CommunicationNode, etc.
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        type: str,
        onchip_energy: float,
        offchip_energy: float,
        runtime: int,
        possible_core_allocation: list[int],
        input_names: list[str],
        output_names: list[str],
        chosen_core_allocation: int | None = None,
    ) -> None:
        """Initialize the Node metaclass

        Args:
            type (str): The type of Node.
            energy (float): The energy consumption of this Node.
            runtime (int): The runtime of this Node.
            possible_core_allocation (int): The core id on which this Node can be mapped.
            inputs: (List[str]): The names of the input tensors of this node
            outpus: (List[str]): The names of the output tensors of this node.
            chosen_core_allocation: The final core allocation of this node
        """
        super().__init__(node_id, node_name)

        self.type = type.lower()
        self.onchip_energy = onchip_energy
        self.offchip_energy = offchip_energy
        self.runtime = runtime
        self.possible_core_allocation = possible_core_allocation
        self.chosen_core_allocation = chosen_core_allocation
        # will be set by the scheduler
        self.start = None
        # will be set by the scheduler
        self.end = None
        # number of data (in bits) only this node consumes (not consumed by any other node)
        self.data_consumed_unique = 0
        # number of data (in bits) only this node produces (not produced by any other node)
        self.data_produced_unique = 0

        self.input_names = input_names
        self.output_names = output_names
        self.offchip_bw = None  # will be set together with the core allocation

    def __str__(self):
        return f"{self.type.capitalize()}Node()"

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

    # def get_possible_core_allocation(self):
    # return self.possible_core_allocation

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
            runtime (int): runtime in cycles
        """
        self.runtime = runtime

    def set_start(self, start: int):
        """Set the start time in cyles of this node.

        Args:
            start (int): start time in cycles
        """
        self.start = start

    def set_end(self, end: int):
        """Set the end time in cycles of this node.

        Args:
            end (int): end time in cycles
        """
        self.end = end

    def set_core_allocation(self, core_allocation: int):
        self.core_allocation = [core_allocation]

    def set_chosen_core_allocation(self, core_allocation: int):
        self.chosen_core_allocation = core_allocation

    def has_end(self) -> bool:
        """Check if this node has already been assigned an end time.

        Returns:
            bool: True if this node has been assigned an end time
        """
        return self.end is not None

    def set_offchip_bandwidth(self, offchip_bw: float):
        self.offchip_bw = offchip_bw
