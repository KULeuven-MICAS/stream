"""The compiled block sizes a fused group can be built at, and the mapping for each.

A kernel declares through ``block_sizes()`` which sizes its compiled source accepts at each
granule position. Two stages need that: the tile search prices a block against a placement,
and the placement stage lays a design out for one. Both read it from here so the rule for
what counts as a candidate is written once.
"""

from dataclasses import replace
from typing import Any

from stream.mapping.mapping import Mapping
from stream.workload.workload import ComputationNode, Workload

CALL_DIMS = ("m", "k", "n")


def block_options(workload: Workload, mapping: Mapping, group) -> dict[Any, tuple[int, ...]]:
    """Compiled block sizes the whole group accepts, per tiling dimension.

    A dimension one kernel compiles in but offers no sizes for is fixed for the group, since
    the kernels either side of it are compiled against the same block. One the group does
    not tile is already at its extent and is not a tiling decision.
    """
    options: dict[Any, set[int]] = {}
    fixed: set[Any] = set()
    for name in group.layers:
        node = workload.get_node_by_name(name)
        if not isinstance(node, ComputationNode) or node not in mapping:
            continue
        kernel = mapping.get(node).kernel
        if kernel is None:
            continue
        dims = workload.get_dims(node)
        offered = kernel.block_sizes()
        for position, _ in kernel.granule():
            dim = dims[position]
            if position in offered:
                options[dim] = options.get(dim, set(offered[position])) & set(offered[position])
            else:
                fixed.add(dim)
    tiled = {dim for dim, _ in group.intra_core_tiling}
    return {
        dim: tuple(sorted(sizes))
        for dim, sizes in options.items()
        if len(sizes) > 1 and dim not in fixed and dim in tiled
    }


def with_block(workload: Workload, mapping: Mapping, gi: int, group, dim, size: int) -> Mapping:
    """The mapping with this group's kernels compiled for ``size`` along ``dim``."""
    mapping = mapping.copy()
    for name in group.layers:
        node = workload.get_node_by_name(name)
        if not isinstance(node, ComputationNode) or node not in mapping:
            continue
        entry = mapping.get(node)
        kernel = entry.kernel
        if kernel is None:
            continue
        dims = workload.get_dims(node)
        fields = {
            CALL_DIMS[position]: size
            for position in kernel.block_sizes()
            if dims[position] == dim and position < len(CALL_DIMS)
        }
        if fields:
            mapping.set(node, replace(entry, kernel=replace(kernel, **fields)))
    extent = workload.get_dimension_size(dim)
    tiling = [(d, min(size, extent) if d == dim else tile) for d, tile in group.intra_core_tiling]
    groups = list(mapping.fused_groups)
    groups[gi] = replace(group, intra_core_tiling=tuple(tiling))
    return mapping.with_fused_groups(groups)
