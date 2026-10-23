"""ZigZag core backend — full operational-array + memory-hierarchy model.

Wraps the external ``zigzag.hardware.architecture.accelerator.Accelerator``
with the same backend protocol used by :class:`AIE2CoreBackend`, so that
both backends are first-class citizens in the framework.
"""

from __future__ import annotations

from typing import Literal

from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.accelerator import Accelerator as _ZigZagAccelerator
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.memory_port import DataDirection, MemoryPort, MemoryPortType

from stream.hardware.ports import (
    OUTPUT,
    READ,
    READ_BY_DATAPATH,
    WRITE,
    WRITE_BY_DATAPATH,
    PortSpec,
    Service,
    input_role,
)

ZIGZAG_DIRECTION_NAMES = {
    DataDirection.RD_OUT_TO_HIGH: READ,
    DataDirection.WR_IN_BY_HIGH: WRITE,
    DataDirection.RD_OUT_TO_LOW: READ_BY_DATAPATH,
    DataDirection.WR_IN_BY_LOW: WRITE_BY_DATAPATH,
}


def operand_role(memory_operand: str) -> str:
    """Neutral role of a ZigZag memory operand: ``O`` is the output, ``I<k>`` input k."""
    if memory_operand == "O":
        return OUTPUT
    if memory_operand.startswith("I") and memory_operand[1:].isdigit():
        return input_role(int(memory_operand[1:]))
    raise ValueError(f"memory operand {memory_operand!r} has no operand role.")


def memory_operand(role: str) -> MemoryOperand:
    """The ZigZag memory operand of a neutral role: the output is ``O``, input k ``I<k>``."""
    if role == OUTPUT:
        return MemoryOperand("O")
    if role.startswith("input") and role[5:].isdigit():
        return MemoryOperand(f"I{role[5:]}")
    raise ValueError(f"role {role!r} has no memory operand.")


class ZigZagCoreBackend(_ZigZagAccelerator):
    """Thin wrapper around the ZigZag ``Accelerator`` that adds the backend protocol.

    Inherits all ZigZag attributes (``operational_array``, ``memory_hierarchy``,
    ``dataflows``, …) and adds ``top_memory_capacity()``, ``get_max_memory_bandwidth()``,
    and ``get_ir()`` so that :class:`~stream.hardware.architecture.core.Core` can
    delegate uniformly without ``isinstance`` checks.
    """

    # ------------------------------------------------------------------
    # Backend protocol — same interface as AIE2CoreBackend
    # ------------------------------------------------------------------

    def top_instances(self) -> frozenset[str]:
        """The core's top-level memory instances: the highest level of each operand."""
        hierarchy = self.memory_hierarchy
        return frozenset(hierarchy.get_memory_levels(op)[-1].name for op in hierarchy.get_operands())

    def top_instance(self, role: str) -> str:
        """The top-level memory instance holding the operand of ``role``; a core with one top-level instance holds
        every operand there, its hierarchy naming the operand or not, as it has nowhere else to."""
        hierarchy, operand = self.memory_hierarchy, memory_operand(role)
        if operand in hierarchy.get_operands():
            return hierarchy.get_memory_levels(operand)[-1].name
        if len(tops := self.top_instances()) == 1:
            (top,) = tops
            return top
        raise ValueError(f"{self} holds {role}'s operand {operand} in none of its top levels {sorted(tops)}")

    def top_memory_capacity(self, instance: str) -> int:
        """Capacity in bits of the top-level memory instance ``instance``, the one the core's memory view binds."""
        hierarchy = self.memory_hierarchy
        tops = {level.name: level for op in hierarchy.get_operands() for level in hierarchy.get_memory_levels(op)[-1:]}
        if instance not in tops:
            raise ValueError(f"{instance} is no top-level memory of {self}: its top levels are {sorted(tops)}")
        return tops[instance].memory_instance.size

    def get_max_memory_bandwidth(self, type: Literal["read"] | Literal["write"]) -> int:
        """Top-level memory read/write bandwidth in bits/cycle."""
        wanted_type = MemoryPortType.READ if type == "read" else MemoryPortType.WRITE
        memory_operand = MemoryOperand("I1")
        ports = self.get_top_memory_instance(memory_operand).ports
        first_port = next((port for port in ports if port.type in (wanted_type, MemoryPortType.READ_WRITE)), None)
        assert first_port is not None, f"{self} does not have a top level memory {type} port."
        return first_port.bw_max

    def port_specs(self) -> tuple[PortSpec, ...]:
        """Ports of each operand's top memory level, with the (role, direction) pairs they serve at that level."""
        hierarchy = self.memory_hierarchy
        tops = {op: len(hierarchy.get_memory_levels(op)) - 1 for op in hierarchy.get_operands()}
        found: dict[tuple[str, str], tuple[MemoryLevel, MemoryPort]] = {}
        serves: dict[tuple[str, str], set[Service]] = {}
        for op in tops:
            level = hierarchy.get_memory_levels(op)[-1]
            for port in level.ports:
                key = (level.name, port.name)
                found.setdefault(key, (level, port))
                serves.setdefault(key, set()).update(
                    (operand_role(str(served)), ZIGZAG_DIRECTION_NAMES[direction])
                    for served, lv, direction in port.served_op_lv_dir
                    if tops.get(served) == lv
                )
        return tuple(
            PortSpec(
                memory=level.name,
                name=port.name,
                bits_per_cycle=float(port.bw_max),
                serves=frozenset(serves[key]),
                read_energy_per_bit=level.memory_instance.r_cost / port.bw_max,
                write_energy_per_bit=level.memory_instance.w_cost / port.bw_max,
            )
            for key, (level, port) in found.items()
        )

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
