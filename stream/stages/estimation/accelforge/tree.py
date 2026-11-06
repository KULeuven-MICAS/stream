"""How a ZigZag compute core nests into the tree of an AccelForge arch, and what its ports cost, worked out without
importing AccelForge: the backend checks the cores it claims with :func:`zigzag_core`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING, Any

from zigzag.hardware.architecture.memory_port import MemoryPortType
from zigzag.hardware.architecture.operational_array import OperationalArray

from stream.stages.estimation.accelforge.workload import PARTIAL_SUMS, RESERVED_NAMES, UnsupportedConversion

if TYPE_CHECKING:
    from zigzag.hardware.architecture.memory_level import MemoryLevel
    from zigzag.hardware.architecture.memory_port import MemoryPort

    from stream.hardware.architecture.backends import ZigZagCoreBackend
    from stream.hardware.architecture.core import Core

PJ = 1e-12
"""Joules per picojoule: Stream energies are pJ per access, AccelForge's J per action."""
CLOCK_HZ = 1.0
"""The clock of arches built from Stream cores: throughputs are per cycle, so AccelForge seconds are cycles."""
UNBOUNDED_SIZE_BITS = 10**10
"""Stream's size for an unbounded memory, as its off-chip and shim cores use; sizes from it up are unbounded."""
COMPUTE = "MAC"


@dataclass(frozen=True)
class Port:
    """One action of a memory: its access width, energy per access, accesses per cycle and the bits one port moves
    per cycle."""

    bits_per_action: float
    pj_per_action: float
    actions_per_cycle: float
    word_bits: float


def zigzag_core(core: Core) -> ZigZagCoreBackend:
    """``core``'s ZigZag hardware, if it converts to an AccelForge arch."""
    refusal = UnsupportedConversion(f"{core}: only zigzag compute cores convert, not {core.core_type}")
    if core.kind != "compute":
        raise refusal
    try:
        backend = core.to_zigzag_core()
    except (TypeError, AttributeError) as error:  # AttributeError: a core built without a hardware backend
        raise refusal from error
    array = backend.operational_array
    if not isinstance(array, OperationalArray):
        raise UnsupportedConversion(f"{core}: in-memory-compute (IMC) arrays have no AccelForge counterpart here")
    if array.total_unit_count == 0:
        raise UnsupportedConversion(f"{core}: its operational array has no units")
    _, _, placed = layout(backend)
    _check_operand_order(core, backend, placed)
    for level in placed:
        if not any(p.type is not MemoryPortType.WRITE for p in level.ports) or not any(
            p.type is not MemoryPortType.READ for p in level.ports
        ):
            raise UnsupportedConversion(f"{level.name}: a memory needs a read and a write port")
    return backend


def layout(zigzag: ZigZagCoreBackend) -> tuple[list[tuple[Any, int]], list[tuple[Any, int]], list[MemoryLevel]]:
    """The array's dimensions, their fanout order outermost first, and the memories in the order the arch places
    them, outer before inner at one depth."""
    dims = list(zigzag.operational_array.dimension_sizes.items())
    levels = list(zigzag.memory_hierarchy.topological_sort())[::-1]  # outermost first
    order = _fanout_order(dims, levels)
    return dims, order, sorted(levels, key=lambda level: depth(level, order))


def _fanout_order(dims: list[tuple[Any, int]], levels: list[MemoryLevel]) -> list[tuple[Any, int]]:
    """The fanout order, outermost first, that copies memories least over dimensions they serve (first such order in
    the array's own order); copies are zero when the memories' served dimensions nest."""

    def copies(order: tuple[tuple[Any, int], ...]) -> int:
        return sum(
            math.prod(size for dim, size in order[: depth(level, order)] if dim in level.served_dimensions) - 1
            for level in levels
        )

    return list(dims if copies(tuple(dims)) == 0 else min(permutations(dims), key=copies))


def depth(level: MemoryLevel, order: list[tuple[Any, int]] | tuple[tuple[Any, int], ...]) -> int:
    """How many fanouts of ``order`` sit above ``level``: all those it does not serve and any before them."""
    unserved = [i for i, (dim, _) in enumerate(order) if dim not in level.served_dimensions]
    return max(unserved) + 1 if unserved else 0


def _check_operand_order(core: Core, zigzag: ZigZagCoreBackend, placed: list[MemoryLevel]) -> None:
    hierarchy = zigzag.memory_hierarchy
    position = {level.name: i for i, level in enumerate(placed)}
    for operand in hierarchy.get_operands():
        chain = [position[level.name] for level in hierarchy.get_memory_levels(operand)]  # innermost first
        if chain != sorted(chain, reverse=True):
            raise UnsupportedConversion(
                f"{core}: operand {operand}'s memories are copied more often further from the array"
            )


def component_name(name: str) -> str:
    """A Stream memory's AccelForge name: prefixed when AccelForge reserves it (``global_*`` variables, set names)."""
    return f"stream_{name}" if name.startswith("global_") or name in RESERVED_NAMES else name


def ports(level: MemoryLevel, r_cost: float, w_cost: float) -> tuple[Port, Port, bool]:
    """The read and write actions of a memory's ports, and whether they are separate ports."""
    reads = [p for p in level.ports if p.type in (MemoryPortType.READ, MemoryPortType.READ_WRITE)]
    writes = [p for p in level.ports if p.type in (MemoryPortType.WRITE, MemoryPortType.READ_WRITE)]
    separate = any(p.type is not MemoryPortType.READ_WRITE for p in level.ports)
    return _port(reads, r_cost), _port(writes, w_cost), separate


def _port(group: list[MemoryPort], cost: float) -> Port:
    """Ports of one type as one action: the first port's access width and energy per bit, all ports' bandwidth."""
    first = group[0]
    actions = sum(p.bw_max for p in group) / first.bw_min
    return Port(first.bw_min, cost * first.bw_min / first.bw_max, actions, first.bw_max)


def one_value_periods(
    port: Port, roles: list[str], one_value: list[str], partial_sum_bits: int | None
) -> tuple[dict[str, str], str | float]:
    """``values_per_action`` and ``throughput_scale`` charging the ``one_value`` roles, whose transfer periods move one
    value into one unit, whole accesses and cycles as ZigZag rounds a period; other roles pay by the bit."""
    bits = {role: _role_bits(role, partial_sum_bits) for role in one_value}
    values = {role: f"1 / {_ceil_div(b, port.bits_per_action)} if {role} else 1" for role, b in bits.items()}
    if one_value != roles or len(roles) != 1 or port.bits_per_action == port.word_bits:
        return values, 1  # with several tensors on one action AccelForge has no per-tensor throughput
    ((role, b),) = bits.items()
    accesses, words = _ceil_div(b, port.bits_per_action), _ceil_div(b, port.word_bits)
    return values, f"{accesses} * {port.bits_per_action} / ({words} * {port.word_bits}) if {role} else 1"


def _role_bits(role: str, partial_sum_bits: int | None) -> str:
    """The bits of a ``role`` value in a memory, as an AccelForge expression: its partial sums' own width if set."""
    if partial_sum_bits and role == "output":
        return f"({partial_sum_bits} if {PARTIAL_SUMS} else output.bits_per_value)"
    return f"{role}.bits_per_value"


def _ceil_div(bits: str, width: float) -> str:
    return f"-(-{bits} // {width})"
