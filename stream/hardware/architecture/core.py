from zigzag.hardware.architecture.accelerator import Accelerator as ZigZagCore


class Core(ZigZagCore):

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
