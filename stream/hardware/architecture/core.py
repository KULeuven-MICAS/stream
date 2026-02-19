from typing import Any, Literal

from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.accelerator import Accelerator as ZigZagCore
from zigzag.hardware.architecture.memory_port import MemoryPortType


class Core(ZigZagCore):
    def __init__(self, args: Any):
        super().__init__(**args)
        self.core_type: str = "zigzag.compute"
        self.type: str = "compute"  # default type for a core
        self.max_object_fifo_depth: int = 16  # default max object FIFO depth for compute
        self.max_buffer_descriptor_depth: int = self.max_object_fifo_depth + 4  # TODO: Define in hardware
        self.utilization: int = 100
        self.row_id: int | None = None
        self.col_id: int | None = None

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

    def get_max_memory_bandwidth(self, type: Literal["read"] | Literal["write"]) -> int:
        """
        Get the top level memory read/write bandwidth of the core in bits/cycle.
        NOTE that this assumes the core has a single top level memory shared across operands.
        NOTE that this uses the first read/write port it finds.
        """
        wanted_type = MemoryPortType.READ if type == "read" else MemoryPortType.WRITE
        memory_operand = MemoryOperand("I1")  # Assuming 'I1' is the top level memory operand
        ports = self.get_top_memory_instance(memory_operand).ports
        first_port = next((port for port in ports if port.type in (wanted_type, MemoryPortType.READ_WRITE)), None)
        assert first_port is not None, f"{self} does not have a top level memory {type} port."
        return first_port.bw_max

    def _get_type_specific_ir(self) -> dict:
        """Return type-specific IR attributes based on ``core_type`` namespace.

        The namespace is the prefix before the first dot in ``core_type``
        (e.g. ``"aie2"`` for ``"aie2.compute"``).  Subclasses or future
        extensions should override or extend this method to expose additional
        per-namespace attributes.
        """
        namespace = self.core_type.split(".")[0] if "." in self.core_type else ""
        if namespace == "aie2":
            return {"max_object_fifo_depth": self.max_object_fifo_depth}
        return {}

    def get_ir(self) -> dict:
        """Return a dictionary representation of this core for serialization.

        The dictionary always contains common fields (id, name, core_type,
        operational array, memory hierarchy, …) and is extended with
        type-specific fields produced by :meth:`_get_type_specific_ir`.
        """
        # --- memory hierarchy ---
        mem_levels: list[dict] = []
        for ml in self.memory_hierarchy.topological_sort():
            bw_max: dict[str, dict[str, int | None]] = {}
            bw_min: dict[str, dict[str, int | None]] = {}
            for op in ml.operands:
                bw_max[str(op)] = {
                    str(direction): bw for direction, bw in ml.bandwidths_max[op].items() if bw is not None
                }
                bw_min[str(op)] = {
                    str(direction): bw for direction, bw in ml.bandwidths_min[op].items() if bw is not None
                }
            mem_levels.append(
                {
                    "name": ml.memory_instance.name,
                    "size_bits": ml.memory_instance.size,
                    "read_cost": ml.memory_instance.r_cost,
                    "write_cost": ml.memory_instance.w_cost,
                    "area": ml.memory_instance.area,
                    "latency": ml.memory_instance.latency,
                    "operands": [str(op) for op in ml.operands],
                    "level_per_operand": {str(k): v for k, v in ml.mem_level_of_operands.items()},
                    "served_dimensions": [str(dim) for dim in ml.served_dimensions],
                    "bandwidths_max": bw_max,
                    "bandwidths_min": bw_min,
                }
            )

        # --- operational array ---
        oa = self.operational_array
        oa_data: dict = {
            "dimension_sizes": {str(k): v for k, v in oa.dimension_sizes.items()},
            "total_unit_count": oa.total_unit_count,
        }
        if hasattr(oa, "unit"):
            oa_data["unit_energy_cost"] = oa.unit.energy_cost
            oa_data["unit_area"] = oa.unit.area

        # --- dataflows ---
        dataflows_ir: dict | None = None
        if self.dataflows is not None:
            dataflows_ir = {
                str(oa_dim): {str(layer_dim): int(factor) for layer_dim, factor in layer_attr.items()}
                for oa_dim, layer_attr in self.dataflows.items()
            }

        d: dict = {
            "id": self.id,
            "name": self.name,
            "core_type": self.core_type,
            "type": self.type,
            "row_id": self.row_id,
            "col_id": self.col_id,
            "utilization": self.utilization,
            "operational_array": oa_data,
            "memory_hierarchy": mem_levels,
            "dataflows": dataflows_ir,
        }

        # Merge type-specific attributes last so they can be easily identified
        d.update(self._get_type_specific_ir())
        return d
