"""The per-core detail of a solved steady state: its axes, each tensor's placement and buffering, and what each tiled
node does on the cores it runs on."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

from stream.opt.allocation.constraint_optimization.report import guarded
from stream.opt.allocation.constraint_optimization.space import core_id
from stream.workload.node import Tensor

if TYPE_CHECKING:
    from stream.allocation.allocation import Allocation
    from stream.datatypes import LayerDim
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.node import ComputationNode, HasIterationSpace, TransferNode
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
    solution = allocation.solution
    return [
        {
            "name": t.name,
            "shape": _shape(t),
            "bits": int(t.size_bits()),
            "loops": loops(ssis),
            "placement": [core.id for core in solution.tensor_placements.get(t, ())],
            "reuse_level": solution.reuse_levels.get(t),
            "depth": solution.depths.get(t),
            "single_buffered": t in solution.single_buffered,
        }
        for t, ssis in allocation.ssis.items()
        if isinstance(t, Tensor)
    ]


def loops(ssis: SteadyStateIterationSpace | None) -> list[dict[str, Any]]:
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
        """Its cores, the output tile at each core position and whether a reader waits on it."""
        tile = self.workload.get_tensor_single_core
        return {
            "dims": [str(dim) for dim in self.workload.get_dims(cn)],
            "loops": loops(self.allocation.ssis.get(cn)),
            "cores": [core.id for core in self.space.core_allocation(cn)[0]],
            "tiles": [_shape(tile(cn.outputs[0], cn, self.mapping, p)) for p in range(self._positions(cn))],
            "waited_on": self.space.is_waited_on(cn),
            **self._sliding(cn),
        }

    def transfer(self, tn: TransferNode) -> dict[str, Any]:
        """Its route, the window of each copy a target holds and what one firing moves to it, the halos its loops
        keep, what each source hands each target, and the bits it moves. A target holds the copies of the nodes it
        runs, each at its position among that copy's targets modulo the positions, as sources pair with targets."""
        space, solution = self.space, self.allocation.solution
        route, n = solution.transfer_routes[tn], self._positions(tn)
        held = [(copy, c) for copy in tn.outputs for c in route.targets if c in space.candidate_cores(copy)]
        held += [(tn.outputs[0], c) for c in route.targets if all(c != target for _, target in held)]
        at = [(copy, [other for other, _ in held[:j]].count(copy) % n) for j, (copy, _) in enumerate(held)]
        moved = [
            self.workload.get_tensor_of_transfer_to_single_core(copy, tn, self.mapping, p, space.ssis.get(copy))
            for copy, p in at
        ]
        return {
            "sources": [core_id(c) for c in route.sources],
            "targets": [core_id(c) for _, c in held],
            "links": [str(link) for link in route.links_used],
            "windows": [_shape(self.workload.get_tensor_single_core(copy, tn, self.mapping, p)) for copy, p in at],
            "moved": [_shape(tile) for tile in moved],
            "halos": [
                {"dim": str(loop.dimension), "type": loop.type.name.lower(), "halo": loop.halo}
                for loop in space.ssis.get(tn.outputs[0]) or ()
                if loop.halo
            ],
            "handed": [
                {"source": core_id(route.sources[i]), "target": core_id(route.targets[j]), "elements": elements}
                for (i, j), elements in space.overlaps(tn).items()
            ],
            "moved_bits": 0 if space.choice_shares_memory(tn, route) else space.moved_bits(tn, route),
            "route_cycles": solution.route_cycles[tn],
            "reuse_factor": solution.reuse_factors.get(tn),
            **self._sliding(tn),
        }

    def _positions(self, node: HasIterationSpace) -> int:
        tiling = self.mapping.get(node).inter_core_tiling
        return prod(f for _, f in tiling[0]) if tiling else 1

    def _sliding(self, node: HasIterationSpace) -> dict[str, Any]:
        """Where a window slides: its work per fused tile along each sliding dim, and how far it runs ahead."""
        return {
            "work": {dim: list(work[node]) for dim, work in self.work.items() if node in work},
            "warmup": self.space.warmup.get(node),
        }
