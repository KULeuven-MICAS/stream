"""ZigZag core backend — full operational-array + memory-hierarchy model.

Wraps the external ``zigzag.hardware.architecture.accelerator.Accelerator``
with the same backend protocol used by :class:`AIE2CoreBackend`, so that
both backends are first-class citizens in the framework.
"""

from __future__ import annotations

from typing import Literal

from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.accelerator import Accelerator as _ZigZagAccelerator
from zigzag.hardware.architecture.memory_port import MemoryPortType


class ZigZagCoreBackend(_ZigZagAccelerator):
    """Thin wrapper around the ZigZag ``Accelerator`` that adds the backend protocol.

    Inherits all ZigZag attributes (``operational_array``, ``memory_hierarchy``,
    ``dataflows``, …) and adds ``get_memory_capacity()``, ``get_max_memory_bandwidth()``,
    and ``get_ir()`` so that :class:`~stream.hardware.architecture.core.Core` can
    delegate uniformly without ``isinstance`` checks.
    """

    # ------------------------------------------------------------------
    # Backend protocol — same interface as AIE2CoreBackend
    # ------------------------------------------------------------------

    def get_memory_capacity(self) -> int:
        """Total top-level memory capacity in bits."""
        memory_operand = MemoryOperand("I1")
        return self.get_top_memory_instance(memory_operand).size

    def get_max_memory_bandwidth(self, type: Literal["read"] | Literal["write"]) -> int:
        """Top-level memory read/write bandwidth in bits/cycle."""
        wanted_type = MemoryPortType.READ if type == "read" else MemoryPortType.WRITE
        memory_operand = MemoryOperand("I1")
        ports = self.get_top_memory_instance(memory_operand).ports
        first_port = next((port for port in ports if port.type in (wanted_type, MemoryPortType.READ_WRITE)), None)
        assert first_port is not None, f"{self} does not have a top level memory {type} port."
        return first_port.bw_max

    def get_ir(self) -> dict:
        """Serialize ZigZag backend-specific fields (operational array,
        memory hierarchy, dataflows) into an IR dict."""

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

        return {
            "operational_array": oa_data,
            "memory_hierarchy": mem_levels,
            "dataflows": dataflows_ir,
        }
