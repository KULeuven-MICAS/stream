"""Placement of a fused group's layers on the AIE compute grid, derived rather than declared.

Every rule is a measurement restated as a derivation:
- Row counts minimize the bottleneck of measured cycles per row, and a layer wider than
  the one it feeds pays the MAC-tiled handover factor its kernel measures -- which is why
  equal width beats a proportionally wider softmax (the two buy the same cycles).
- A linear chain short enough to stack takes a row per layer of the same columns, so
  every handover crosses adjacent tiles' shared memory; any other group takes disjoint
  column tenancies sized by the same bottleneck rule, or the whole array alone.
- A bandwidth-bound layer (its call finishes before its operands could move) gets one
  core per column on the row beside the memory tile: further rows only wait on the DMA.
- The shared parallel dimension splits over the widest column count it divides in whole
  granules.
"""

from dataclasses import dataclass
from itertools import product

from stream.stages.estimation.kernel_cycles import MEASURED_KERNEL_CYCLES, calls_ops

# The MAC-tiled handover variant of the softmax measures 2.09x its row-major body; any
# layer handing core-to-core to a narrower consumer pays it, which is what makes width
# and kernel speed substitutes rather than adding up.
TILED_HANDOVER_FACTOR = 2.09

# Measured throughputs the anchors imply, for kernels whose exact symbol has no anchor:
# a mixed scale (one layer anchored, its neighbour on a hand-fed percentage) skews every
# proportional rule, so the fallback uses the same family rates the anchors measure.
GEMM_MACS_PER_CYCLE = 151.0
ELEMENTWISE_OPS_PER_CYCLE = 16.0
# One tile's DMA moves this many bf16 elements per cycle.
DMA_ELEMENTS_PER_CYCLE = 32.0


@dataclass(frozen=True)
class LayerPlan:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    split: tuple[tuple[int, int], ...]
    by_row: bool


def layer_cost(kernel) -> float:
    """Cycles one call of this layer's kernel takes, measured where an anchor exists."""
    measured = MEASURED_KERNEL_CYCLES.get(getattr(kernel, "function_name", None))
    if measured is not None:
        return measured
    ops = calls_ops(kernel) or 1
    return ops / (GEMM_MACS_PER_CYCLE if len(kernel.granule()) > 2 else ELEMENTWISE_OPS_PER_CYCLE)


def bandwidth_bound(kernel) -> bool:
    """Whether one call finishes before its operands could cross a DMA."""
    granule = kernel.granule()
    if len(granule) > 2:
        return False
    operands = max(2, len(kernel.operand_layouts() or ()))
    move_cycles = operands * (calls_ops(kernel) or 0) / DMA_ELEMENTS_PER_CYCLE
    return layer_cost(kernel) <= move_cycles


def row_counts(costs: list[float], num_rows: int) -> tuple[int, ...]:
    """Rows per layer minimizing the bottleneck of cost per row, handover factor included."""
    best: tuple[tuple[float, int], tuple[int, ...]] | None = None
    for counts in product(range(1, num_rows + 1), repeat=len(costs)):
        if sum(counts) > num_rows:
            continue
        eff = [
            cost * (TILED_HANDOVER_FACTOR if i + 1 < len(counts) and r > counts[i + 1] else 1.0) / r
            for i, (cost, r) in enumerate(zip(costs, counts))
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


def column_budgets(costs: list[float], num_columns: int) -> tuple[int, ...]:
    """Columns per layer, disjoint and exhaustive, minimizing the bottleneck per column."""
    best: tuple[float, tuple[int, ...]] | None = None
    for counts in product(range(1, num_columns + 1), repeat=len(costs)):
        if sum(counts) != num_columns:
            continue
        key = max(c / n for c, n in zip(costs, counts))
        if best is None or key < best[0]:
            best = (key, counts)
    if best is None:
        return tuple(1 for _ in costs)
    return best[1]
