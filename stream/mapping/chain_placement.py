"""Placement of a fused group's layers on the AIE compute grid, derived rather than declared."""

from itertools import product
from math import prod

TILED_HANDOVER_FACTOR = 2.09
DMA_ELEMENTS_PER_CYCLE = 32.0
MIN_CALL_OPERANDS = 2


def call_sizes(kernel, node) -> dict[int, int]:
    """The size of one call along each dimension of ``node`` the kernel library declares."""
    return {position: size for position, size, _ in kernel.call_tile(node)}


def is_matmul(kernel) -> bool:
    return kernel.spec.family == "matmul"


def layer_cost(kernel) -> float:
    """Cycles one call of this layer's kernel takes, measured where the library has a call."""
    shape = kernel.call_shape()
    ops = prod(shape.values())
    if measured := kernel.spec.call_cycles(shape):
        cycles, measured_ops = measured
        return cycles * ops / measured_ops
    return ops / kernel.library.families[kernel.spec.family].ops_per_cycle


def bandwidth_bound(kernel) -> bool:
    """Whether one call finishes before its operands could cross a DMA."""
    if is_matmul(kernel):
        return False
    operands = max(MIN_CALL_OPERANDS, len(kernel.operand_layouts() or ()))
    move_cycles = operands * prod(kernel.call_shape().values()) / DMA_ELEMENTS_PER_CYCLE
    return layer_cost(kernel) <= move_cycles


def row_counts(
    costs: list[float],
    num_rows: int,
    state_consumers: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Rows per layer minimizing the bottleneck of cost per row, handover factor included."""
    best: tuple[tuple[float, int], tuple[int, ...]] | None = None
    for counts in product(range(1, num_rows + 1), repeat=len(costs)):
        if sum(counts) > num_rows:
            continue
        if any(i > 0 and counts[i] > counts[i - 1] for i in state_consumers):
            continue
        eff = [
            cost * (TILED_HANDOVER_FACTOR if i + 1 < len(counts) and r > counts[i + 1] else 1.0) / r
            for i, (cost, r) in enumerate(zip(costs, counts, strict=True))
        ]
        key = (max(eff), sum(counts))
        if best is None or key < best[0]:
            best = (key, counts)
    assert best is not None
    return best[1]


def widest_columns(extent: int, granule: int, lanes: int, num_columns: int) -> int:
    """The widest column count the shared dimension still divides in whole granules."""
    for columns in range(num_columns, 0, -1):
        if extent % (granule * columns * lanes) == 0:
            return columns
    return 1


def column_budget_options(
    costs: list[float], num_columns: int, caps: list[int] | None = None, limit: int = 2
) -> list[tuple[int, ...]]:
    """The ``limit`` best column budgets per layer, minimizing the bottleneck per column."""
    caps = caps or [num_columns] * len(costs)
    ranked: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
    for counts in product(range(1, num_columns + 1), repeat=len(costs)):
        if sum(counts) != num_columns:
            continue
        key = tuple(sorted((c / min(n, cap) for c, n, cap in zip(costs, counts, caps, strict=True)), reverse=True))
        ranked.append((key, counts))
    ranked.sort()
    return [counts for _, counts in ranked[:limit]]
