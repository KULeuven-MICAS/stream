"""ZigZag compute cores as AccelForge arches, with Stream's own energies and a 1 Hz clock so AccelForge's seconds are
Stream's cycles; a memory operand maps to a rename (``I1``: ``input``, ``I2``: ``weight``, ``O``: ``output``). See
``docs/source/accelforge.md`` for what does not carry over."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import stream.stages.estimation.accelforge.library  # noqa: F401

# isort: split
from accelforge import Spec
from accelforge.frontend.arch import Action, Arch, Compute, Container, Memory, Spatial, TensorHolderAction, Tensors
from stream.datatypes import ELEMENT_BITS
from stream.stages.estimation.accelforge.tree import (
    CLOCK_HZ,
    COMPUTE,
    PJ,
    UNBOUNDED_SIZE_BITS,
    Port,
    component_name,
    depth,
    layout,
    one_value_periods,
    ports,
    zigzag_core,
)
from stream.stages.estimation.accelforge.workload import NOTHING, PARTIAL_SUMS, ROLE_OF_OPERAND

if TYPE_CHECKING:
    from zigzag.hardware.architecture.memory_level import MemoryLevel

    from accelforge.frontend.workload import Workload
    from stream.hardware.architecture.backends import ZigZagCoreBackend
    from stream.hardware.architecture.core import Core

ROLES = " | ".join(ROLE_OF_OPERAND.values())


def explicit_spec(arch: Arch, workload: Workload) -> Spec:
    """A spec of ``arch`` and ``workload`` that uses only the arch's explicit values, never a component model."""
    spec = Spec(arch=arch, workload=workload)
    spec.config.component_models = []
    spec.config.use_installed_component_models = False
    return spec


def core_to_arch(core: Core, *, free_memory_order: bool = False) -> Arch:
    """The AccelForge arch of a ZigZag compute core: memories outermost first, a fanout per array dimension below the
    memories serving it, a MAC compute leaf. An operand's top memory is unbounded, as ZigZag's LOMA treats it;
    ``free_memory_order`` lets different tensors' storage nest in any order, as ZigZag does."""
    zigzag = zigzag_core(core)
    array = zigzag.operational_array
    dims, order, placed = layout(zigzag)
    accumulator = (getattr(core, "operand_precision", None) or {}).get("accumulator")
    nodes: list[Memory | Container | Compute] = []
    memories: list[Memory] = []  # root first
    for fanouts_above in range(len(order) + 1):
        for level in (level for level in placed if depth(level, order) == fanouts_above):
            copies = math.prod(size for dim, size in order[:fanouts_above] if dim in level.served_dimensions)
            units = math.prod(size for dim, size in dims if dim in level.served_dimensions)
            memory = _memory(zigzag, level, copies, units, accumulator)
            memory.tensors.force_memory_hierarchy_order = not free_memory_order
            nodes.append(memory)
            memories.append(memory)
        if fanouts_above < len(order):
            nodes.append(_fanout(*order[fanouts_above]))
    outermost = memories[0]  # it keeps every tensor no rename names too, such as a third input
    outermost.tensors.keep = f"{outermost.tensors.keep} | ~({ROLES})"
    nodes.append(_compute(COMPUTE, array.unit.energy_cost * PJ, CLOCK_HZ))
    return Arch(nodes=nodes)


def _memory(zigzag: ZigZagCoreBackend, level: MemoryLevel, copies: int, units: int, accumulator: str | None) -> Memory:
    """A memory serving ``units`` array units per instance, ``copies`` times over dimensions it should serve."""
    hierarchy = zigzag.memory_hierarchy
    operands = [str(operand) for operand in level.operands]
    tops = {str(op) for op in level.operands if hierarchy.get_memory_levels(op)[-1] is level}
    instance = level.memory_instance
    read, write, separate = ports(level, instance.r_cost, instance.w_cost)
    size = math.inf if tops or instance.size >= UNBOUNDED_SIZE_BITS else instance.size
    if copies > 1:
        size = size // copies if math.isfinite(size) else size
        read, write = (
            Port(p.bits_per_action, p.pj_per_action, p.actions_per_cycle / copies, p.word_bits) for p in (read, write)
        )
    bits_per_value = {}
    if accumulator and "O" in operands and "O" not in tops:
        bits_per_value = {PARTIAL_SUMS: ELEMENT_BITS[accumulator]}
    roles = [ROLE_OF_OPERAND[op] for op in operands]
    innermost = {str(op) for op in level.operands if hierarchy.get_memory_levels(op)[0] is level}
    one_value = [ROLE_OF_OPERAND[op] for op in operands if units == 1 and op in innermost]
    partial_sum_bits = bits_per_value.get(PARTIAL_SUMS)
    return Memory(
        name=component_name(level.name),
        size=size,
        leak_power=0,
        area=0,
        separate_read_write_ports=separate,
        bits_per_value=bits_per_value,
        actions=[
            _memory_action(name, port, *one_value_periods(port, roles, one_value, partial_sum_bits))
            for name, port in (("read", read), ("write", write))
        ],
        tensors=Tensors(
            keep=" | ".join(roles) or NOTHING,
            may_keep=NOTHING,
        ),
    )


def _memory_action(
    name: str, port: Port, values_per_action: Mapping[str, str] | None = None, throughput_scale: str | float = 1
) -> TensorHolderAction:
    return TensorHolderAction(
        name=name,
        energy=port.pj_per_action * PJ,
        throughput=port.actions_per_cycle * CLOCK_HZ,
        throughput_scale=throughput_scale,
        latency=0,
        bits_per_action=port.bits_per_action,
        values_per_action=dict(values_per_action or {}),
    )


def _fanout(dim: object, size: int) -> Container:
    """A fanout of ``size`` along the operational-array dimension ``dim``."""
    spatial = Spatial(name=str(dim), fanout=size)
    return Container(name=f"array_{dim}", spatial=[spatial])


def _compute(name: str, energy_j: float, throughput: float) -> Compute:
    return Compute(
        name=name,
        leak_power=0,
        area=0,
        actions=[Action(name="compute", energy=energy_j, throughput=throughput, latency=0)],
    )
