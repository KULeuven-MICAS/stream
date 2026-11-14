"""What share of a node's work each of the cores it is placed on actually performs."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
from xdsl.ir.affine import AffineDimExpr

from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.workload.node import ComputationNode
from stream.workload.utils import affine_bounds, affine_coefficients
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
    workload: Workload, mapping: Mapping, node: ComputationNode, one: Core, other: Core, fusion_splits: dict
) -> bool:
    """Whether two cores do the same amount of this node's work."""
    steps = split_steps(workload, mapping, node, fusion_splits)
    return core_work_share(workload, mapping, node, one, steps) == core_work_share(
        workload, mapping, node, other, steps
    ) and last_tile_shares(workload, mapping, node, one, fusion_splits) == last_tile_shares(
        workload, mapping, node, other, fusion_splits
    )


def output_extent_diff(workload: Workload, node: ComputationNode, fusion_splits: dict) -> dict[int, int]:
    """Per dim of ``node`` indexing an output axis alone, how far the output ends past (negative: short of) the dim's
    whole extent, where the two differ."""
    unique_dims = workload.unique_dimensions()[0]
    whole = [workload.get_dimension_size(z) * fusion_splits.get(z, 1) for z in unique_dims]
    dims, output = workload.get_dims(node), node.outputs[0]
    edges: dict[int, int] = {}
    for expr, extent in zip(node.get_mapping(output).results, output.subview.source.type.get_shape(), strict=True):
        if isinstance(expr, AffineDimExpr):
            low, high = affine_bounds(dims[expr.position], whole)
            if edge := extent - (high - low + 1):
                edges[expr.position] = edge
    return edges


def last_tile_shares(
    workload: Workload, mapping: Mapping | None, node: ComputationNode, core: Core, fusion_splits: dict
) -> dict[int, tuple[int, float]]:
    """Per dim of :func:`output_extent_diff`, if ``core`` is the last along it: the edge and the share of the core's
    tiles that make it (the last along the dim)."""
    edges = output_extent_diff(workload, node, fusion_splits)
    unique_dims = workload.unique_dimensions()[0]
    position = _core_position(workload, mapping, node, core) if edges else {}
    shares: dict[int, tuple[int, float]] = {}
    for p, edge in edges.items():
        coefficients = affine_coefficients(workload.get_dims(node)[p], len(unique_dims))[1]
        along = [z for z, c in zip(unique_dims, coefficients, strict=True) if c]
        if all(index == cores - 1 for z, (index, cores) in position.items() if z in along):
            shares[p] = (edge, 1 / prod(fusion_splits.get(z, 1) for z in along))
    return shares


def _core_position(
    workload: Workload, mapping: Mapping | None, node: ComputationNode, core: Core
) -> dict[LayerDim, tuple[int, int]]:
    """Along each unique dim ``node``'s split cuts evenly, the index of ``core`` and the number of cores, the last dim
    of the split fastest."""
    cores = _query_split(mapping, node) if mapping is not None and node in mapping else None
    tiling = workload.get_unique_dims_inter_core_tiling(node, mapping) if cores else ()
    factors = [factor for _, factor in tiling]
    if not cores or core not in cores or cores.index(core) >= prod(factors):
        return {}
    index = np.unravel_index(cores.index(core), factors)
    return {
        z: (int(i), factor)
        for (z, factor), i in zip(tiling, index, strict=True)
        if workload.get_dimension_size(z) % factor == 0
    }


def computed_fraction(workload: Workload, mapping: Mapping, node: ComputationNode, core: Core, steps: int = 1) -> float:
    """Fraction of this core's steps that compute anything, the rest being skipped."""
    cores = _query_split(mapping, node)
    if cores is None or len(cores) <= 1 or core not in cores:
        return 1.0
    return min(1.0, core_work_share(workload, mapping, node, core, steps) * len(cores))
