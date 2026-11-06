"""A node, described as an Einsum, mapped on a core with AccelForge's single-Einsum mapper, and the best mapping read
into a :class:`~.backend.MappedNode`."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from stream.stages.estimation.accelforge.library import contained

# isort: split
from accelforge import Metrics, Spec
from accelforge.frontend.mapping import Compute, Loop, MappingNode, Nested, Reservation, Spatial, Storage
from accelforge.frontend.workload import Einsum, TensorAccess, Workload
from accelforge.mapper.FFM import map_workload_to_arch
from accelforge.model.main import evaluate_mapping
from stream.stages.estimation.accelforge.arch import core_to_arch, explicit_spec
from stream.stages.estimation.accelforge.backend import MappedNode
from stream.stages.estimation.accelforge.workload import rank_variables

if TYPE_CHECKING:
    from accelforge.frontend.arch import Arch
    from accelforge.mapper.FFM import Mappings
    from stream.hardware.architecture.core import Core
    from stream.stages.estimation.accelforge.workload import EinsumDescription

SPATIAL_ABOVE_TEMPORAL = "a spatial loop above a temporal loop above a storage node"


class LoopTrips(NamedTuple):
    spatial: bool
    trips: dict[int, int]
    """Per dim position the loop iterates over, its trips: the dim's tile above the loop over its own tile."""


def to_workload(einsum: EinsumDescription) -> Workload:
    """An AccelForge workload of ``einsum`` alone."""
    accesses = [
        TensorAccess(name=a.name, projection=dict(a.projection), output=a.output, bits_per_value=a.bits_per_value)
        for a in einsum.accesses
    ]
    variables = rank_variables(einsum.name, einsum.sizes)
    shape = [f"0 <= {variable} < {size}" for variable, size in zip(variables, einsum.sizes, strict=True)]
    return Workload(
        einsums=[
            Einsum(
                name=einsum.name, tensor_accesses=accesses, iteration_space_shape=shape, renames=dict(einsum.renames)
            )
        ]
    )


def map_node(einsum: EinsumDescription, core: Core, memory_order: str, imperfect_spatial_loops: bool) -> MappedNode:
    """The best mapping of ``einsum`` on ``core``. In ``free`` memory order, a node whose free-order template
    AccelForge rejects is mapped again in ``forced`` order."""
    with contained():
        return _map(einsum, core, memory_order, imperfect_spatial_loops)


def _map(einsum: EinsumDescription, core: Core, memory_order: str, imperfect_spatial_loops: bool) -> MappedNode:
    workload = to_workload(einsum)
    if memory_order == "forced":
        return _mapped(einsum, core, workload, imperfect_spatial_loops, "forced", None)
    try:
        return _mapped(einsum, core, workload, imperfect_spatial_loops, "free", None)
    except (ValueError, AssertionError) as error:
        if not free_order_refusal(error):
            raise
        cause = " ".join(str(error).split("Mapping:")[0].split())
        fallback = f"free order refused, mapped in forced order: {type(error).__name__}: {cause}"
        return _mapped(einsum, core, workload, imperfect_spatial_loops, "forced", fallback)


def free_order_refusal(error: BaseException) -> bool:
    """Whether ``error`` is AccelForge 1.0.521 rejecting one of its own free-order templates while it splits a mapping
    per tensor (seen on tpu_like, tpu_v7_mxu and meta_prototype)."""
    message = " ".join(str(error).split())
    return (isinstance(error, ValueError) and SPATIAL_ABOVE_TEMPORAL in message) or (
        isinstance(error, AssertionError) and message == "BUG"
    )


def _mapped(
    einsum: EinsumDescription,
    core: Core,
    workload: Workload,
    imperfect_spatial_loops: bool,
    memory_order: str,
    fallback: str | None,
) -> MappedNode:
    arch = core_to_arch(core, free_memory_order=memory_order == "free")
    start = time.perf_counter()
    result = map_best(explicit_spec(arch, workload), imperfect_spatial_loops)
    wall_s = time.perf_counter() - start
    path = _einsum_path(result.mapping().nodes, einsum.name)
    loops = _loop_trips(path, einsum)
    actions = result.actions(per_einsum=True, per_component=True, per_tensor=True)
    return MappedNode(
        latency=float(result.latency(per_einsum=True)[einsum.name]),
        energy_j=float(result.energy(per_einsum=True)[einsum.name]),
        actions={(c, str(t), a): float(v) for (e, c, t, a), v in actions.items() if e == einsum.name},
        bits_per_action=_bits_per_action(arch),
        spatial_unrolling=_spatial_unrolling(loops),
        temporal_cycles=math.prod(max(loop.trips.values()) for loop in loops if not loop.spatial),
        top_storage=_top_storage(path),
        units=math.prod(int(spatial.fanout) for node in arch.nodes for spatial in getattr(node, "spatial", ())),
        mapping=tuple(node.compact_str() for node in path if not isinstance(node, Reservation)),
        memory_order=memory_order,
        fallback=fallback,
        wall_s=wall_s,
    )


def map_best(spec: Spec, imperfect_spatial_loops: bool) -> Mappings:
    """The detailed evaluation of the mapping with the fewest cycles, then the least energy, as ZigZag's
    ``MinimalLatencyStage`` picks. Evaluating only that one, unlike ``Spec.map_workload_to_arch``, writes no
    ``mapping.svg``."""
    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
    spec.mapper.explore_imperfect_spatial_loops = imperfect_spatial_loops
    pareto = map_workload_to_arch(spec, eval_in_detail=False, print_progress=False)
    if len(pareto) == 0:
        raise ValueError("AccelForge found no valid mapping")
    totals = pareto.data
    best = min(
        range(len(pareto)),
        key=lambda i: (float(totals["Total<SEP>latency"].iloc[i]), float(totals["Total<SEP>energy"].iloc[i])),
    )
    detailed = copy.deepcopy(spec)
    detailed.model.metrics = detailed.mapper.info_metrics
    detailed.mapping = pareto.mapping(best)
    return evaluate_mapping(detailed, flattened_arches=pareto.flattened_arches, evaluated_specs=pareto.evaluated_specs)


def _einsum_path(nodes: Sequence[MappingNode], einsum: str) -> list[MappingNode]:
    """The mapping nodes from the root down to ``einsum``'s compute node; a one-Einsum mapping nests without splits."""
    path: list[MappingNode] = []
    for node in nodes:
        if isinstance(node, Nested):
            return [*path, *_einsum_path(node.nodes, einsum)]
        if isinstance(node, Compute) and node.einsum == einsum:
            return [*path, node]
        path.append(node)
    raise LookupError(f"no compute node of Einsum {einsum} in the mapping")


def _loop_trips(path: Sequence[MappingNode], einsum: EinsumDescription) -> list[LoopTrips]:
    """Each loop on ``path`` over the Einsum's dims, each partial last trip counted whole, where AccelForge counts it
    by fraction."""
    positions = {variable: i for i, variable in enumerate(rank_variables(einsum.name, einsum.sizes))}
    tiles = dict(enumerate(einsum.sizes))
    loops: list[LoopTrips] = []
    for node in path:
        if not isinstance(node, Loop):
            continue
        names = node.rank_variable if isinstance(node.rank_variable, set | frozenset) else {node.rank_variable}
        if dims := [positions[name] for name in names if name in positions]:
            tile = int(node.tile_shape)
            loops.append(LoopTrips(isinstance(node, Spatial), {dim: math.ceil(tiles[dim] / tile) for dim in dims}))
            tiles.update(dict.fromkeys(dims, tile))
    return loops


def _spatial_unrolling(loops: list[LoopTrips]) -> dict[int, int]:
    """How many units the spatial loops spread each of the Einsum's dims (by position) over."""
    unrolled: dict[int, int] = {}
    for loop in loops:
        for dim, trips in loop.trips.items():
            if loop.spatial and trips > 1:
                unrolled[dim] = unrolled.get(dim, 1) * trips
    return unrolled


def _top_storage(path: Sequence[MappingNode]) -> dict[str, str]:
    top: dict[str, str] = {}
    for node in path:
        if isinstance(node, Storage):
            for tensor in node.tensors:
                top.setdefault(str(tensor), str(node.component))
    return top


def _bits_per_action(arch: Arch) -> dict[tuple[str, str], float]:
    return {
        (str(node.name), str(action.name)): float(action.bits_per_action)
        for node in arch.nodes
        if hasattr(node, "tensors")
        for action in node.actions
    }
