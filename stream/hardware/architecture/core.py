from typing import Any

from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.accelerator import Accelerator as ZigZagCore


class Core(ZigZagCore):
    def __init__(self, args: Any):
        super().__init__(**args)
        self.type = "compute"  # default type for a core

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Core)
            and self.id == other.id
            and self.operational_array == other.operational_array
            and self.memory_hierarchy == other.memory_hierarchy
            and self.dataflows == other.dataflows
        )

    def has_same_performance(self, other: "Core") -> bool:  # type: ignore
        return (
            self.operational_array == other.operational_array
            and self.memory_hierarchy.has_same_performance(other.memory_hierarchy)
            and self.dataflows == other.dataflows
        )

    def __hash__(self) -> int:
        return self.id

    @staticmethod
    def from_zigzag_core(core: ZigZagCore) -> "Core":
        core.__class__ = Core
        return core  # type: ignore

    def get_memory_capacity(self) -> int:
        """
        Get the total memory capacity of the core in bits.
        NOTE that this assumes the core has a single top level memory shared across operands.
        """
        memory_operand = MemoryOperand("I1")  # Assuming 'I1' is the top level memory operand
        return self.get_top_memory_instance(memory_operand).size
