"""The per-core detail of a solved steady state: its axes, each tensor's placement and buffering, and what each tiled
node does on the cores it runs on."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

from stream.opt.allocation.constraint_optimization.families.traffic import link_name
from stream.opt.allocation.constraint_optimization.report import guarded, reported_reuse
from stream.opt.allocation.constraint_optimization.utils import absent_trips
from stream.workload.demand import Box, DemandItem, covered, firing_count
from stream.workload.node import ComputationNode, Tensor, TransferNode

if TYPE_CHECKING:
    from stream.allocation.allocation import Allocation
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.datatypes import LayerDim
    from stream.hardware.architecture.memory import Memory
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.node import HasIterationSpace
    from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace


def steady_state_detail(allocation: Allocation, space: DecisionSpace) -> dict[str, Any]:
    """The steady state's iteration count, axes and tensors, and per tiled node name its detail; a node whose
    detail cannot be built is left out."""
    sliding = {loop.dimension for ssis in space.ssis.values() for loop in ssis if loop.halo}
    detail = _NodeDetail(allocation, space, sliding)
    workload = allocation.problem.workload
    nodes = {
        node.name: guarded(f"steady-state detail of {node.name}", build, node)
        for build, group in (
            (detail.compute, workload.get_computation_nodes()),
            (detail.transfer, workload.get_transfer_nodes()),
        )
        for node in group
    }
    return {
        "iterations": allocation.problem.iterations,
        "axes": _axes(allocation, sliding),
        "tensors": _tensors(allocation),
        "nodes": {name: found for name, found in nodes.items() if found is not None},
    }


def _axes(allocation: Allocation, sliding: set[LayerDim]) -> list[dict[str, Any]]:
    workload, splits = allocation.problem.workload, allocation.problem.fusion_splits
    return [
        {
            "dim": str(dim),
            "size": workload.get_dimension_size(dim),
            "split": splits.get(dim, 1),
            "sliding": dim in sliding,
        }
        for dim in workload.unique_dimensions()[0]
    ]


def _tensors(allocation: Allocation) -> list[dict[str, Any]]:
    """Each tensor as the solution placed and buffered it, a transfer's copy at the window its reader holds."""
    solution, workload, mapping = allocation.solution, allocation.problem.workload, allocation.problem.mapping
    writers = {copy: tn for tn in workload.get_transfer_nodes() for copy in tn.outputs}
    held = {t: workload.get_tensor_single_core(t, tn, mapping) for t, tn in writers.items() if t in allocation.ssis}
    return [
        {
            "name": t.name,
            "shape": _shape(held.get(t, t)),
            "bits": int(held.get(t, t).size_bits()),
            "loops": _loops(ssis),
            "placement": [memory.name for memory in solution.tensor_placements.get(t, ())],
            "reuse_level": solution.reuse_levels.get(t),
            "depth": solution.depths.get(t),
            "single_buffered": t in solution.single_buffered,
        }
        for t, ssis in allocation.ssis.items()
        if isinstance(t, Tensor)
    ]


def _loops(ssis: SteadyStateIterationSpace | None) -> list[dict[str, Any]]:
    """The loops of a steady-state iteration space, outermost first."""
    return [
        {
            "dim": str(iv.dimension),
            "size": iv.size,
            "type": iv.type.name.lower(),
            "effect": iv.effect.name.lower(),
            "reuse": iv.reuse.name.lower(),
            "halo": iv.halo,
        }
        for iv in reversed(ssis.variables if ssis else ())
    ]


def _shape(tensor: Tensor) -> list[int]:
    return [int(s) for s in tensor.shape]


def _per_firing(points: int, spanned: int) -> int | float:
    """Points over the firings a demand spans, an integer where they divide."""
    return points // spanned if points % spanned == 0 else points / spanned


class _NodeDetail:
    """What one tiled node does on its cores, read off the solution and the decision space it was solved in."""

    def __init__(self, allocation: Allocation, space: DecisionSpace, sliding: set[LayerDim]) -> None:
        self.allocation, self.space = allocation, space
        problem = allocation.problem
        self.workload, self.mapping = problem.workload, problem.mapping
        self.work = {
            str(dim): self.workload.get_sliding_work(dim, n)
            for dim, n in problem.fusion_splits.items()
            if dim in sliding
        }

    def compute(self, cn: ComputationNode) -> dict[str, Any]:
        """Its cores, how often it runs, the output tile at each core position and whether a reader waits on it."""
        tile = self.workload.get_tensor_single_core
        return {
            "dims": [str(dim) for dim in self.workload.get_dims(cn)],
            "loops": _loops(self.allocation.ssis.get(cn)),
            "calls": self.allocation.problem.iterations // absent_trips(self.space.ssis[cn]),
            "cores": [core.id for core in self.space.cores(cn)],
            "tiles": [_shape(tile(cn.outputs[0], cn, self.mapping, p)) for p in range(self._positions(cn))],
            "waited_on": self.space.is_waited_on(cn),
            **self._sliding(cn),
        }

    def transfer(self, tn: TransferNode) -> dict[str, Any]:
        """Its loops, route, copies and the copy each target holds (the copy of a node the target runs), the window
        and moved tile of that copy, each copy's halos, the demand each source serves at each target, the bits it moves
        and the iterations one firing serves."""
        space, solution = self.space, self.allocation.solution
        route = solution.transfer_routes[tn]
        # Pair each copy with the targets it may land in; a target no copy may land in gets the first copy.
        held = [(copy, m) for copy in tn.outputs for m in route.targets if m in space.candidate_memories(copy)]
        held += [(tn.outputs[0], c) for c in route.targets if all(c != target for _, target in held)]
        at = [
            (copy, [other for other, _ in held[:j]].count(copy) % self._copy_positions(tn, copy))
            for j, (copy, _) in enumerate(held)
        ]
        moved = [
            self.workload.get_tensor_of_transfer_to_single_core(copy, tn, self.mapping, p, space.ssis.get(copy))
            for copy, p in at
        ]
        return {
            "loops": _loops(self.allocation.ssis.get(tn)),
            "sources": [m.name for m in route.sources],
            "targets": [m.name for _, m in held],
            "copy_names": [copy.name for copy in tn.outputs],
            "target_copies": [tn.outputs.index(copy) for copy, _ in held],
            "links": [link_name(link) for link in route.links_used],
            "windows": [_shape(self.workload.get_tensor_single_core(copy, tn, self.mapping, p)) for copy, p in at],
            "moved": [_shape(tile) for tile in moved],
            "halos": [
                {"dim": str(loop.dimension), "type": loop.type.name.lower(), "halo": loop.halo, "copy": i}
                for i, copy in enumerate(tn.outputs)
                for loop in reversed(space.ssis[copy].variables)
                if loop.halo
            ],
            "demand": self._demand_by_kind(tn, route),
            **self._copy_bits(tn),
            "link_bits": self._link_bits(tn, route),
            "route_cycles": solution.route_cycles[tn],
            **self._sliding(tn),
        }

    def _copy_bits(self, tn: TransferNode) -> dict[str, Any]:
        """Per copy, the bits one of its firings moves and its reuse factor, a box several copies read charged to the
        one firing most often; the transfer's own are its most often firing copy's reuse and bits per such firing."""
        space, solution = self.space, self.allocation.solution
        route = solution.transfer_routes[tn]
        reuse = [
            reported_reuse(space.reuse_levels[(copy, solution.reuse_levels[copy])], space.ssis[tn])
            if copy in solution.reuse_levels
            else reported_reuse(solution.reuse_factors[tn], space.ssis[tn])
            for copy in tn.outputs
        ]
        moved = [0.0] * len(tn.outputs)
        if not space.choice_shares_memory(tn, route):
            for copies in space.copy_sets(tn, route):
                first = min(copies, key=lambda c: reuse[c])
                moved[first] += space.moved_bits(tn, route, copies=copies)
        top = min((r for r, bits in zip(reuse, moved, strict=True) if bits), default=reuse[0])
        return {
            "moved_bits": sum(bits * top / r for bits, r in zip(moved, reuse, strict=True)),
            "reuse_factor": top,
            "copy_moved_bits": moved,
            "copy_reuse_factors": reuse,
        }

    def _demand_by_kind(self, tn: TransferNode, route: MulticastPathPlan) -> list[dict[str, Any]]:
        """Per source, target and kind, the elements of the target's demand the source serves per firing: over links
        ('moved'), from a source in or sharing the target's memory ('in_place'), from a larger copy in that memory
        ('local_copy'), or as a source's partial sum of a split reduction ('partial_sum')."""
        served: dict[tuple[Memory, Memory, str], list[Box]] = {}
        kind = "in_place" if self.space.choice_shares_memory(tn, route) else "moved"
        sent = [item for item in route.delivered_items if not item.reduce]
        delivered = {(item.target, item.box, item.copies) for item in sent}
        for item in sent:
            served.setdefault((route.sources[item.sources[0]], route.targets[item.target], kind), []).append(item.box)
        for item in route.demand:
            for key in self._served_without_links(route, item, delivered):
                served.setdefault(key, []).append(item.box)
        spanned = firing_count(route.demand)
        return [
            {"source": one.name, "target": other.name, "kind": kind, "elements": _per_firing(covered(boxes), spanned)}
            for (one, other, kind), boxes in served.items()
        ]

    def _served_without_links(
        self, route: MulticastPathPlan, item: DemandItem, delivered: set[tuple[int, Box, tuple[int, ...]]]
    ) -> list[tuple[Memory, Memory, str]]:
        """The source that serves a demand item no link carries, and the kind."""
        target = route.targets[item.target]
        if item.reduce:
            return [(route.sources[s], target, "partial_sum") for s in item.sources]
        if not item.sources:
            return [(target, target, "local_copy")]
        if (item.target, item.box, item.copies) in delivered:
            return []
        held = [s for s in item.sources if route.sources[s] is target] or [
            s for s in item.sources if self.space.hardware.shares_memory(route.sources[s], target)
        ]
        return [(route.sources[(held or list(item.sources))[0]], target, "in_place")]

    def _link_bits(self, tn: TransferNode, route: MulticastPathPlan) -> list[float]:
        """Per link of the route, the bits one firing carries over it, as its route cycles count them."""
        if self.space.choice_shares_memory(tn, route):
            return [0.0] * len(route.links_used)
        moved, shares = self.space.moved_bits(tn, route), dict(route.link_shares)
        return [moved * shares.get(link, 0.0) if shares else moved for link in route.links_used]

    def _positions(self, node: HasIterationSpace) -> int:
        tiling = self.mapping.get(node).inter_core_tiling
        return prod(f for _, f in tiling[0]) if tiling else 1

    def _copy_positions(self, tn: TransferNode, copy: Tensor) -> int:
        """The positions a copy's tile is told apart by: those of the computation reading it, of the transfer where
        another transfer reads it, one for an output edge."""
        reader = next(node for node in self.workload.successors(tn) if copy in node.inputs)
        if isinstance(reader, ComputationNode):
            return self._positions(reader)
        return self._positions(tn) if isinstance(reader, TransferNode) else 1

    def _sliding(self, node: HasIterationSpace) -> dict[str, Any]:
        """Where a window slides: its work per fused tile along each sliding dim, and how far it runs ahead."""
        return {
            "work": {dim: list(work[node]) for dim, work in self.work.items() if node in work},
            "warmup": self.space.warmup.get(node),
        }
