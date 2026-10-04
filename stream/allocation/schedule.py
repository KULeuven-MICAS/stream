from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

from stream.hardware.architecture.core import Core
from stream.workload.node import Tensor
from stream.workload.steady_state.iteration_space import (
    IterationVariableType,
    LoopEffect,
    Reuse,
    SteadyStateIterationSpace,
)

if TYPE_CHECKING:
    from stream.allocation.problem import SteadyStateProblem
    from stream.allocation.solution import AllocationSolution
    from stream.cost_model.core_cost_lut import CoreCostLUT
    from stream.datatypes import LayerDim
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.mapping.mapping import Mapping
    from stream.workload.node import ComputationNode, HasIterationSpace
    from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

IterationSpaces = dict["HasIterationSpace | Tensor", SteadyStateIterationSpace]

#: Nest depth of each steady-state loop kind, outermost first.
_LOOP_NEST_DEPTH: dict[str, int] = {
    "temporal": 0,
    "spatiotemporal": 1,
    "spatial": 2,
    "core_temporal": 3,
    "core_spatial": 4,
    "kernel": 5,
}


@dataclass(frozen=True)
class SteadyStateSchedule:
    """A solved steady state as downstream reads it: the problem it solves, the mapping and iteration
    spaces the solution decided, the constraint families it was built from with their options, and the
    solution itself."""

    problem: SteadyStateProblem
    mapping: Mapping
    ssis: IterationSpaces
    backend: str
    families: tuple[tuple[str, dict[str, Any]], ...]
    solution: AllocationSolution

    @property
    def source_workload(self) -> Workload:
        """The fused group's workload before its transfers were made explicit."""
        return self.problem.source_workload

    @property
    def workload(self) -> Workload:
        """The steady-state workload, with its transfers."""
        return self.problem.workload

    @property
    def iterations(self) -> int:
        return self.problem.iterations

    @property
    def fusion_splits(self) -> dict[LayerDim, int]:
        return self.problem.fusion_splits

    @property
    def accelerator(self) -> Accelerator:
        return self.problem.accelerator

    @property
    def cost_lut(self) -> CoreCostLUT:
        return self.problem.cost_lut

    @property
    def cost_to_rank(self) -> float:
        """What two solved designs should be compared by: the latency objective the solve minimised first."""
        return self.solution.primary_cost

    @property
    def estimated_cycles(self) -> float:
        """What running this steady state takes: a lone node pipelines to its throughput bound,
        while a fused group is held to how its nodes overlap, which the solved cost captures."""
        if len(self.source_workload.get_computation_nodes()) == 1:
            return self.solution.throughput_bound
        return self.cost_to_rank

    def get_ir(self) -> dict:
        """The schedule as a plain dict: latencies, solve configuration, statistics and families, fusion splits,
        mapping, performance report and the steady-state inspection view."""
        stats = self.solution.solve_stats
        latency = self.solution.latency
        return {
            "latency": {
                "total": latency.total,
                "per_iteration": latency.per_iteration,
                "overlap_between_iterations": latency.overlap,
                "fill": latency.fill,
            },
            "backend": self.backend,
            "solve": {
                "status": stats.status,
                "solver": stats.solver,
                "mip_gap": stats.mip_gap,
                "objective": stats.objective,
                "solve_time_s": stats.solve_time_s,
                "node_count": stats.node_count,
                "iteration_count": stats.iteration_count,
            },
            "families": [{"name": name, "options": options} for name, options in self.families],
            "fusion_splits": {str(dim): size for dim, size in self.fusion_splits.items()},
            "mapping": self.mapping.get_ir(),
            "performance": self.solution.performance,
            "steady_state": self._steady_state_ir(),
        }

    def _core_loops(self, cn: ComputationNode) -> list[dict]:
        """The loop nest inside one core (ZigZag mapping), as ``core_*`` loops; empty for a non-ZigZag core."""
        # Resolve by name -- the mapping is keyed by steady-state nodes, the cost LUT by the costed node.
        try:
            lut_node = next(n for n in self.cost_lut.get_nodes() if n.name == cn.name)
            allocation = self.mapping.get(lut_node).resource_allocation
            cores = [c for slot in (allocation or ()) for c in slot if isinstance(c, Core)]
            if not cores:
                return []
            entry = self.cost_lut.get_cost(lut_node, cores[0])
        except Exception:  # noqa: BLE001
            return []
        mapping = getattr(entry, "mapping", None)
        if mapping is None:
            return []

        loops: list[dict] = []

        def add(dim: str, size: int, kind: str) -> None:
            # No de-dup: ZigZag splits one dim over several levels, so equal-size loops are real levels.
            if int(size) > 1:
                loops.append({"dim": dim, "size": int(size), "type": kind, "node": cn.name})

        # ZigZag annotates the nest once per operand; take one operand's view (summing multiplies every dim).
        def one_operand(per_operand: dict) -> list:
            return next(iter(per_operand.values()), [])

        # Array unrollings first: these run in parallel, so they sit outside the temporal walk.
        for level in one_operand(getattr(mapping.spatial_mapping, "mapping_dict_origin", {})):
            for layer_dim, size in level:
                add(str(layer_dim), size, "core_spatial")
        for level in one_operand(getattr(mapping.temporal_mapping, "mapping_dic_stationary", {})):
            for layer_dim, size in level:
                add(str(layer_dim), size, "core_temporal")
        return loops

    def _steady_state_ir(self) -> dict | None:
        """Serialise the tiled/steady-state inspection view (operators, loop nest, transfer graph); None on failure."""
        try:
            operators = [
                {
                    "name": cn.name,
                    "op": getattr(cn, "type", "computation"),
                    "tensors": [{"name": t.name, "shape": [int(s) for s in t.shape]} for t in cn.tensors],
                }
                for cn in self.source_workload.get_computation_nodes()
            ]
            # The for-loop nest over the steady-state iteration space (deduped across operands, size > 1).
            loops: list[dict] = []
            seen: set = set()
            for ssis in self.ssis.values():
                for iv in ssis.variables:
                    # ABSENT: the node lacks the dim (unrolling replicates it); counting it double-counts one unrolling.
                    if iv.effect is LoopEffect.ABSENT:
                        continue
                    key = (str(iv.dimension), int(iv.size))
                    if int(iv.size) > 1 and key not in seen:
                        seen.add(key)
                        loops.append({"dim": str(iv.dimension), "size": int(iv.size), "type": iv.type.name.lower()})
            # Below the tile: expand each node's intra-core mapping per node (fused groups stay separate).
            expanded = False
            for cn in self.source_workload.get_computation_nodes():
                core_loops = self._core_loops(cn)
                for loop in core_loops:
                    loop["node"] = cn.name
                loops.extend(core_loops)
                expanded = expanded or bool(core_loops)
            if expanded:
                # Drop the kernel stand-in once expanded (it would double-count the intra-core work).
                loops = [loop for loop in loops if loop["type"] != "kernel"]

            def _nest_order(loop: dict) -> tuple[str, int]:
                return loop.get("node") or "", _LOOP_NEST_DEPTH.get(loop["type"], len(_LOOP_NEST_DEPTH))

            loops.sort(key=_nest_order)
            # The tiled workload graph WITH transfer nodes -- the tensor copies that reside on-chip.
            tiled_nodes: list[dict] = []
            for cn in self.workload.get_computation_nodes():
                tiled_nodes.append({"name": cn.name, "kind": "compute", "op": getattr(cn, "type", "computation")})
            for tn in self.workload.get_transfer_nodes():
                out = tn.outputs[0] if tn.outputs else None
                transfer_type = getattr(tn, "transfer_type", None)
                tiled_nodes.append(
                    {
                        "name": tn.name,
                        "kind": "transfer",
                        "transfer_type": getattr(transfer_type, "name", None),
                        "tensor": out.name if out is not None else None,
                        "elements": int(prod(out.shape)) if out is not None else 0,
                    }
                )
            edges = [{"source": s.name, "target": t.name} for s, t in self.workload.edges()]
            return {"operators": operators, "loops": loops, "tiled_graph": {"nodes": tiled_nodes, "edges": edges}}
        except Exception as exc:  # noqa: BLE001 -- inspection view must never break a solved run
            logger.warning("could not build steady-state IR: %s", exc)
            return None


def solved_iteration_spaces(
    workload: Workload, ssis: IterationSpaces, reuse_levels: dict[Tensor, int]
) -> IterationSpaces:
    """Copies of ``ssis`` whose temporal loops carry the reuse the solution chose for each tensor, mirrored
    onto the transfers that move it."""
    solved = {key: SteadyStateIterationSpace(tuple(copy(iv) for iv in space.variables)) for key, space in ssis.items()}
    for t, space in solved.items():
        if isinstance(t, Tensor):
            assert t in reuse_levels, f"Tensor {t.name} does not have a reuse level assigned."
            for i, iv in enumerate(space.get_applicable_temporal_variables()):
                iv.reuse = Reuse.REUSE if i <= reuse_levels[t] else Reuse.NO_REUSE
    # Propagate spatial reuse across transfer boundaries: when one side of a
    # transfer has a SPATIAL variable that is represented as a SPATIOTEMPORAL on the
    # other side (same dimension and size), mark that spatiotemporal as REUSE so that
    # both endpoints display the same reuse boundary.
    for node in workload.get_transfer_nodes():
        for src in node.inputs:
            for dst in node.outputs:
                _propagate_spatial_reuse(solved, src, dst)
                _propagate_spatial_reuse(solved, dst, src)
    # Mirror solved reuse from the moved tensor's SSIS (priced) onto each transfer's SSIS, by (dim, size).
    for node in workload.get_transfer_nodes():
        governing = next((t for t in (*node.outputs, *node.inputs) if isinstance(t, Tensor) and t in solved), None)
        if governing is None:
            continue
        reuse_by_loop = {(v.dimension, v.size): v.reuse for v in solved[governing].get_temporal_variables()}
        for iv in solved[node].get_temporal_variables():
            if (iv.dimension, iv.size) in reuse_by_loop:
                iv.reuse = reuse_by_loop[(iv.dimension, iv.size)]
    return solved


def _propagate_spatial_reuse(ssis: IterationSpaces, spatial_side: Tensor, temporal_side: Tensor) -> None:
    """Mark spatiotemporal variables on ``temporal_side`` as REUSE when they
    match (dimension, size) of an applicable spatial variable on ``spatial_side``."""
    if spatial_side not in ssis or temporal_side not in ssis:
        return
    spatial_keys_not_in_temporal = {
        (iv.dimension, iv.size)
        for iv in ssis[spatial_side].variables
        if iv.type == IterationVariableType.SPATIAL
        and iv.applicable
        and iv not in ssis[temporal_side].variables  # only look at temporal side vars that are not spatial
    }
    if not spatial_keys_not_in_temporal:
        return
    seen_spatial_keys = set()
    for iv in ssis[temporal_side].variables:
        is_spatiotemporal = iv.type in (IterationVariableType.SPATIOTEMPORAL,)
        match = (iv.dimension, iv.size) in spatial_keys_not_in_temporal
        not_seen = (iv.dimension, iv.size) not in seen_spatial_keys
        if is_spatiotemporal and match and not_seen:
            if iv.applicable:  # Only set to reuse if it's applicable
                iv.reuse = Reuse.REUSE
            seen_spatial_keys.add((iv.dimension, iv.size))


def solved_mapping(workload: Workload, mapping: Mapping, solution: AllocationSolution) -> Mapping:
    """A copy of ``mapping`` with each transfer bound to the route and memory cores the solution chose."""
    solved = mapping.copy()
    for tr, route in solution.transfer_routes.items():
        solved.set_for_node(
            tr,
            resource_allocation=(route,),
            inter_core_tiling=tuple(),
            memory_allocation=solution.memory_cores.get(tr, tuple()),
        )
    for tr in workload.get_transfer_nodes():
        assert len(solved.get(tr).resource_allocation) == 1, (
            f"Transfer node {tr.name} should have exactly one resource allocation after update."
        )
    return solved
