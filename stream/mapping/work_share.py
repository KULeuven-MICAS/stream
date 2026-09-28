"""What share of a node's work each of the cores it is placed on actually performs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.workload.node import ComputationNode
from stream.workload.workload import Workload

if TYPE_CHECKING:
    from stream.compiler.kernels.aie_kernel import AIEKernel


def _query_split(mapping: Mapping, node: ComputationNode) -> tuple[Core, ...] | None:
    """The cores this node's first inter-core split spreads it over."""
    entry = mapping.get(node)
    if not entry.resource_allocation or not entry.inter_core_tiling or not entry.inter_core_tiling[0]:
        return None
    return tuple(entry.resource_allocation[0])


def _group_kernels(workload: Workload, mapping: Mapping, node: ComputationNode) -> list[AIEKernel]:
    """Every kernel in the fused group holding this node, or the node's own kernel outside a group."""
    group = next((g for g in mapping.fused_groups if node.name in g.layers), None)
    if group is None:
        names = [node.name]
    else:
        present = {n.name for n in workload.node_list}
        names = [name for name in group.layers if name in present]
    kernels = []
    for name in names:
        other = workload.get_node_by_name(name)
        if other in mapping and mapping.get(other).kernel is not None:
            kernels.append(mapping.get(other).kernel)
    return kernels


def split_steps(workload: Workload, mapping: Mapping, node: ComputationNode, fusion_splits: dict) -> int:
    """How many slices of its split dimension one core ends up holding."""
    slots = mapping.get(node).inter_core_tiling if node in mapping else ()
    if not slots or not slots[0]:
        return 1
    dims = workload.get_dims(node)
    position = slots[0][0][0].position
    if position >= len(dims):
        return 1
    return max(int(fusion_splits.get(dims[position], 1)), 1)


def core_work_share(workload: Workload, mapping: Mapping, node: ComputationNode, core: Core, steps: int = 1) -> float:
    """Fraction of ``node``'s rectangular work that ``core`` performs."""
    cores = _query_split(mapping, node)
    if cores is None:
        return 1.0
    width = len(cores)
    if width <= 1 or core not in cores:
        return 1.0 / max(width, 1)
    index = cores.index(core)
    uniform = 1.0 / width
    shares = [k.work_share(index, width, steps) for k in _group_kernels(workload, mapping, node)]
    return max(shares, key=lambda share: abs(share - uniform), default=uniform)


def interchangeable(
    workload: Workload, mapping: Mapping, node: ComputationNode, one: Core, other: Core, steps: int
) -> bool:
    """Whether two cores do the same amount of this node's work."""
    return core_work_share(workload, mapping, node, one, steps) == core_work_share(
        workload, mapping, node, other, steps
    )


def computed_fraction(workload: Workload, mapping: Mapping, node: ComputationNode, core: Core, steps: int = 1) -> float:
    """Fraction of this core's steps that compute anything, the rest being skipped."""
    cores = _query_split(mapping, node)
    if cores is None or len(cores) <= 1 or core not in cores:
        return 1.0
    return min(1.0, core_work_share(workload, mapping, node, core, steps) * len(cores))
