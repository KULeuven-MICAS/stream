"""The compiled block sizes a fused group can be built at, and the mapping for each."""

from dataclasses import replace

from stream.datatypes import LayerDim
from stream.mapping.mapping import FusedGroup, Mapping
from stream.workload.workload import ComputationNode, Workload


def _group_kernels(workload: Workload, mapping: Mapping, group: FusedGroup):
    for name in group.layers:
        node = workload.get_node_by_name(name)
        if isinstance(node, ComputationNode) and node in mapping and mapping.get(node).kernel is not None:
            yield node, mapping.get(node)


def block_options(workload: Workload, mapping: Mapping, group: FusedGroup) -> dict[LayerDim, tuple[int, ...]]:
    """Block sizes the whole group accepts per tiled dimension, from the kernels' compiled block lists."""
    options: dict[LayerDim, set[int]] = {}
    floors: dict[LayerDim, int] = {}
    fixed: set[LayerDim] = set()
    for node, entry in _group_kernels(workload, mapping, group):
        dims = workload.get_dims(node)
        for position, _, call_dim in entry.kernel.call_tile():
            dim = dims[position]
            if call_dim.blocks:
                options[dim] = options.get(dim, set(call_dim.blocks)) & set(call_dim.blocks)
            elif call_dim.divisor:
                floors[dim] = max(floors.get(dim, 1), call_dim.divisor)
            elif not call_dim.runtime:
                fixed.add(dim)
    tiled = {dim for dim, _ in group.intra_core_tiling}
    narrowed = {
        dim: tuple(sorted(size for size in sizes if size % floors.get(dim, 1) == 0))
        for dim, sizes in options.items()
        if dim not in fixed and dim in tiled
    }
    return {dim: sizes for dim, sizes in narrowed.items() if len(sizes) > 1}


def with_block(workload: Workload, mapping: Mapping, gi: int, group: FusedGroup, dim: LayerDim, size: int) -> Mapping:
    """The mapping with this group's kernels compiled for ``size`` along ``dim``."""
    mapping = mapping.copy()
    for node, entry in _group_kernels(workload, mapping, group):
        dims = workload.get_dims(node)
        fields = {
            call_dim.name: size
            for position, _, call_dim in entry.kernel.call_tile()
            if dims[position] == dim and (call_dim.blocks or call_dim.divisor)
        }
        if fields:
            kernel = replace(entry.kernel, **fields)
            kernel.validate()
            mapping.set(node, replace(entry, kernel=kernel))
    tiling = [(d, size if d == dim else tile) for d, tile in group.intra_core_tiling]
    groups = list(mapping.fused_groups)
    groups[gi] = replace(group, intra_core_tiling=tuple(tiling))
    return mapping.with_fused_groups(groups)
