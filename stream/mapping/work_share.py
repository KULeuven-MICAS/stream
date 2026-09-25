"""What share of a node's work each of the cores it is placed on actually performs.

A node split N ways is normally assumed to give each core an N-th of the work, and for a
rectangular iteration space that is exact. It is not exact when the space is not
rectangular: the cores then do unequal amounts, latency is set by the busiest one, and a
model that divides by N sees none of it.

The machinery to express that already exists. The cost model prices a node per (node,
core) and the allocator takes the max over cores, so a node only has to say how its work
is really spread and everything downstream follows. Uniform is the default and reproduces
a plain split exactly.

Today the one uneven space is causal attention, where the key extent a query block attends
grows with its position. Another source of unevenness is another branch in
:func:`core_work_share`, not a new concept.
"""

from typing import Any

UNIFORM = None


def _query_split(mapping: Any, node: Any) -> tuple[tuple[Any, ...], int] | None:
    """The cores this node is split over, and the position of the dimension split."""
    entry = mapping.get(node)
    slots = entry.resource_allocation
    tiling = entry.inter_core_tiling
    if not slots or not tiling or not tiling[0]:
        return None
    return tuple(slots[0]), tiling[0][0][0].position


def _group_kernels(workload: Any, mapping: Any, node: Any) -> list[Any]:
    """Every kernel in the fused group holding this node.

    A fused group steps in lockstep, so an uneven iteration space anywhere in it is uneven
    for all of it: the softmax and the value accumulation skip exactly where the causal
    score GEMM does.
    """
    for group in mapping.fused_groups:
        if node.name not in group.layers:
            continue
        kernels = []
        for name in group.layers:
            try:
                other = workload.get_node_by_name(name)
            except Exception:  # noqa: BLE001 -- a group may name a node this sub-workload lacks
                continue
            if other in mapping and mapping.get(other).kernel is not None:
                kernels.append(mapping.get(other).kernel)
        return kernels
    kernel = mapping.get(node).kernel if node in mapping else None
    return [kernel] if kernel is not None else []


def split_steps(workload: Any, mapping: Any, node: Any, fusion_splits: dict) -> int:
    """How many slices of its split dimension one core ends up holding."""
    slots = mapping.get(node).inter_core_tiling if node in mapping else ()
    if not slots or not slots[0]:
        return 1
    dims = workload.get_dims(node)
    position = slots[0][0][0].position
    if position >= len(dims):
        return 1
    return max(int(fusion_splits.get(dims[position], 1)), 1)


def interchangeable(workload: Any, mapping: Any, node: Any, one: Any, other: Any, steps: int) -> bool:
    """Whether two cores do the same amount of this node's work.

    Identical hardware is not enough: on an uneven iteration space two identical cores
    holding different slices do different amounts, so a cost measured for one is not a cost
    for the other.
    """
    return core_work_share(workload, mapping, node, one, steps) == core_work_share(
        workload, mapping, node, other, steps
    )


def core_work_share(workload: Any, mapping: Any, node: Any, core: Any, steps: int = 1) -> float:
    """Fraction of ``node``'s rectangular work that ``core`` performs.

    ``steps`` is how many times the split dimension comes round, so a core holding one
    slice per step holds ``steps`` of them in total.
    """
    split = _query_split(mapping, node)
    if split is None:
        return 1.0
    cores, _ = split
    width = len(cores)
    if width <= 1 or core not in cores:
        return 1.0 / max(width, 1)
    index = cores.index(core)
    uniform = 1.0 / width
    # A fused group steps in lockstep, so it is as uneven as its most uneven member. Taking
    # the largest share would be the opposite: a rectangular member returns 1/width and,
    # since an uneven share is below uniform for every core but the busiest, would floor
    # the whole group flat. Take the member that departs furthest from uniform instead.
    shares = [k.work_share(index, width, steps) for k in _group_kernels(workload, mapping, node)]
    return max(shares, key=lambda share: abs(share - uniform), default=uniform)


def computed_fraction(workload: Any, mapping: Any, node: Any, core: Any, steps: int = 1) -> float:
    """Fraction of this core's steps that compute anything, the rest being skipped.

    The share is of the whole rectangle, so a core that skipped nothing would hold its
    uniform 1/width of it; what it holds against that is how many of its steps it computes.
    A step it skips still waits for the operands it would have used, which is why the
    allocator wants this number and not only the share.
    """
    split = _query_split(mapping, node)
    if split is None or len(split[0]) <= 1 or core not in split[0]:
        return 1.0
    share = core_work_share(workload, mapping, node, core, steps)
    return min(1.0, share * len(split[0]))


def uniform_share(splits: int) -> float:
    return 1.0 / max(splits, 1)


__all__ = ["computed_fraction", "core_work_share", "uniform_share"]
