"""The compiled block sizes a fused group can be built at, and the mapping for each."""

from dataclasses import replace
from math import prod

from stream.datatypes import LayerDim
from stream.mapping.mapping import FusedGroup, Mapping
from stream.workload.workload import ComputationNode, Workload

ACCUMULATOR_BITS = {"fp32": 32}
"""Bits per element of the running sum a kernel keeps in addition to its output, by its ``ACCUMULATOR`` format."""


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
        for position, _, call_dim in entry.kernel.call_tile(node):
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
            for position, _, call_dim in entry.kernel.call_tile(node)
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


def _measured_kernel(workload: Workload, mapping: Mapping, group: FusedGroup):
    """The group's one kernel node, its mapping entry and its library entry, where that entry measures calls; None
    otherwise."""
    kernels = list(_group_kernels(workload, mapping, group))
    if len(kernels) != 1:
        return None
    node, entry = kernels[0]
    spec = entry.kernel.spec if getattr(entry.kernel, "library", None) is not None else None
    return (node, entry, spec) if spec is not None and spec.cycles else None


def measured_calls(workload: Workload, mapping: Mapping, group: FusedGroup) -> list[dict[LayerDim, int]]:
    """The call shapes the group's kernel library measures, as the tile each puts on the group's dims: those of a
    group with one kernel that has measured calls, other than the shape it is built at, where the kernel compiles the
    shape and its tiles divide the dims."""
    found = _measured_kernel(workload, mapping, group)
    if found is None:
        return []
    node, entry, spec = found
    dims = workload.get_dims(node)
    by_name = {call_dim.name: dims[position] for position, _, call_dim in entry.kernel.call_tile(node)}
    current = {call_dim.name: size for _, size, call_dim in entry.kernel.call_tile(node)}
    shapes = []
    for measured, _ in spec.cycles:
        if set(measured) != set(by_name) or measured == current:
            continue
        try:
            spec.validate(measured)
        except ValueError:
            continue
        if all(workload.get_dimension_size(by_name[name]) % size == 0 for name, size in measured.items()):
            shapes.append({by_name[name]: size for name, size in measured.items()})
    return shapes


def with_call(
    workload: Workload, mapping: Mapping, gi: int, group: FusedGroup, call: dict[LayerDim, int]
) -> Mapping | None:
    """The mapping with this group's kernel compiled for the ``call`` tile on its dims and the group's innermost tile
    of each such dim taking it; the dim's larger tiles stay. None where the call does not divide one of them."""
    tiling = list(group.intra_core_tiling)
    for dim, size in call.items():
        at = next((i for i, (d, _) in enumerate(tiling) if d == dim), None)
        if at is None:
            continue
        if any(d == dim and i != at and tile % size for i, (d, tile) in enumerate(tiling)):
            return None
        tiling[at] = (dim, size)
    called = mapping
    for dim, size in call.items():
        called = with_block(workload, called, gi, called.fused_groups[gi], dim, size)
    groups = list(called.fused_groups)
    groups[gi] = replace(group, intra_core_tiling=tuple(tiling))
    return called.with_fused_groups(groups)


def call_bound(  # noqa: PLR0913
    workload: Workload, mapping: Mapping, group: FusedGroup, link_bits: float, capacity_bits: float = float("inf")
) -> float | None:
    """Cycles per MAC a core of the group spends at least per call: the call's measured cycles, or the nearest
    measured call's scaled to its size, or where longer, the time the operand tiles it fetches anew take over a
    ``link_bits``-wide link into the core. A tile is fetched once per run of the innermost loops that do not index it
    and that nest inside a larger tile of their dim. Infinite where the core cannot hold two of each operand tile, an
    accumulator per output tile it holds across those loops where its kernel sums in one, and its output tile within
    ``capacity_bits``. None for a group without one kernel with measured calls."""
    found = _measured_kernel(workload, mapping, group)
    if found is None:
        return None
    node, entry, spec = found
    dims = workload.get_dims(node)
    call = {dims[position]: size for position, size, _ in entry.kernel.call_tile(node)}
    cycles, ops = spec.call_cycles({call_dim.name: size for _, size, call_dim in entry.kernel.call_tile(node)})
    macs = prod(call.values())
    tiling = group.intra_core_tiling
    streamed, held = 0.0, 0.0
    for tensor in node.inputs:
        indexed = {dim for dim, steps in workload.strides_for_tensor(tensor).items() if any(steps)}
        bits = prod(size for dim, size in call.items() if dim in indexed) * tensor.operand_type.bitwidth
        held += 2 * bits
        for dim, tile in tiling:
            larger = sorted(t for d, t in tiling if d == dim and t > tile)
            if dim in indexed or not larger:
                break
            bits /= larger[0] // tile
        streamed += bits
    output = node.outputs[0]
    produced = {dim for dim, steps in workload.strides_for_tensor(output).items() if any(steps)}
    tile = prod(size for dim, size in call.items() if dim in produced)
    tiles = 1
    for dim, size in tiling:
        larger = sorted(t for d, t in tiling if d == dim and t > size)
        if dim not in produced or not larger:
            break
        tiles *= larger[0] // size
    accumulator = ACCUMULATOR_BITS.get(getattr(entry.kernel, "ACCUMULATOR", None), 0)
    held += tile * (accumulator * tiles + output.operand_type.bitwidth)
    if held > capacity_bits:
        return float("inf")
    return max(cycles * macs / ops, streamed / link_bits) / macs
