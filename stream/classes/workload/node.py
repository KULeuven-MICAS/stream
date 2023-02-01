from abc import ABCMeta
from typing import List

class Node(metaclass=ABCMeta):
    """Abstract base class that represents a piece of an algorithmic workload.
    Example: ComputationNode, CommunicationNode, etc.
    """

    def __init__(self, type: str, energy: float, runtime: int, core_allocation: int, input_names: List[str], output_names: List[str]) -> None:
        """Initialize the Node metaclass

        Args:
            type (str): The type of Node.
            energy (float): The energy consumption of this Node.
            runtime (int): The runtime of this Node.
            core_allocation (int): The core id on which this Node is mapped.
            inputs: (List[str]): The names of the input tensors of this node
            outpus: (List[str]): The names of the output tensors of this node. 
        """
        self.type = type.lower()
        self.energy = energy
        self.runtime = runtime
        self.core_allocation = core_allocation
        self.start = None  # will be set by the scheduler
        self.end = None  # will be set by the scheduler
        self.data_consumed_unique = 0  # number of data (in bits) only this node consumes (not consumed by any other node)
        self.data_produced_unique = 0  # number of data (in bits) only this node produces (not produced by any other node)

        self.input_names = input_names
        self.output_names = output_names

    def __str__(self):
        return f"{self.type.capitalize()}Node()"

    def __repr__(self):
        return str(self)

    def get_energy(self):
        """Get the energy of running this node
        """
        return self.energy

    def get_runtime(self):
        """Get the runtime of running this node.
        """
        return self.runtime

    def get_start(self):
        """Get the start time in cycles of this node.
        """
        return self.start

    def get_end(self):
        """Get the end time in cycles of this node.
        """
        return self.end

    def get_core_allocation(self):
        return self.core_allocation

    def set_energy(self, energy):
        """Set the energy of running this node.

        Args:
            energy (float): energy consumption of this node
        """
        self.energy = energy

    def set_runtime(self, runtime):
        """Set the runtime of running this node.

        Args:
            runtime (int): runtime in cycles
        """
        self.runtime = runtime

    def set_start(self, start):
        """Set the start time in cyles of this node.

        Args:
            start (int): start time in cycles
        """
        self.start = start

    def set_end(self, end):
        """Set the end time in cycles of this node.

        Args:
            end (int): end time in cycles
        """
        self.end = end

    def set_core_allocation(self, core_allocation):
        self.core_allocation = core_allocation

    def has_end(self) -> bool:
        """Check if this node has already been assigned an end time.

        Returns:
            bool: True if this node has been assigned an end time
        """
        return self.end is not None
