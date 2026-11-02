import logging
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from stream.cost_model.roofline import hardware_roofline, run_roofline
from stream.hardware.architecture.core import Core
from stream.ir.allocation import AllocationIR
from stream.ir.roofline import RooflineIR
from stream.mapping.mapping import (
    check_offchip_capacity,
    offchip_bits,
    placement_options,
    with_cut_landings,
    with_held_placements,
)
from stream.opt.allocation.constraint_optimization.report import guarded
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.node import edge_tensor

if TYPE_CHECKING:
    from stream.allocation.allocation import Allocation
    from stream.hardware.architecture.memory import Memory
    from stream.workload.tensor import Tensor
    from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


def _compute_columns(allocation: "Allocation") -> tuple[int, ...]:
    """The distinct compute-tile columns the solved allocation occupies."""
    columns: set[int] = set()
    for node_mapping in allocation.mapping.values():
        for group in node_mapping.resource_allocation or ():
            items = group if isinstance(group, (list, tuple)) else (group,)
            for item in items:
                if isinstance(item, Core) and item.type == "compute" and item.col_id is not None:
                    columns.add(item.col_id)
    return tuple(sorted(columns))


class FusionGroupIterationStage(Stage):
    """Iterate over fusion groups, running the inner pipeline once per group on its own workload and mapping, beside
    what the off-chip memories hold. A group places the inputs the mapping does not where it reads them first, and the
    groups after it read them there."""

    reads = ("accelerator", "output_path", "sub_workloads", "sub_mappings")
    optional_reads = ("memory_accesses",)
    writes = ("workload", "mapping", "output_path", "group_index", "offchip_held")
    result_reads = ("allocation",)
    result_writes = (
        "total_latency",
        "group_latencies",
        "group_columns",
        "group_cycles",
        "group_wall_times",
        "group_allocations",
        "group_memory_accesses",
        "roofline",
    )

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.get("accelerator")
        self.output_path = self.ctx.get("output_path")
        self.sub_workloads = self.ctx.get("sub_workloads")
        self.sub_mappings = self.ctx.get("sub_mappings")

    def run(self):  # noqa: PLR0915
        sub_workloads = self.sub_workloads
        total_latency = 0.0
        group_latencies: dict[int, float] = {}
        group_columns: dict[int, tuple[int, ...]] = {}
        group_cycles: dict[int, float] = {}
        group_wall_times: dict[int, float] = {}
        group_allocations: dict[int, dict | None] = {}
        group_memory_accesses: dict[int, dict | None] = {}
        final_ctx = None

        assert len(sub_workloads) == len(self.sub_mappings), (
            f"Mismatch: {len(sub_workloads)} sub-workloads vs {len(self.sub_mappings)} sub-mappings"
        )
        self.sub_mappings = with_cut_landings(self.accelerator, sub_workloads, self.sub_mappings)
        check_offchip_capacity(self.accelerator, sub_workloads, self.sub_mappings)

        for i, sub_workload in enumerate(sub_workloads):
            group_output = os.path.join(self.output_path, f"group_{i}")
            os.makedirs(group_output, exist_ok=True)

            self.ctx.set(
                workload=sub_workload,
                mapping=self.sub_mappings[i],
                output_path=group_output,
                group_index=i,
                offchip_held=self._held_beside(sub_workloads, i),
            )

            logger.info(f"Running inner pipeline for group {i} ({len(sub_workload.get_computation_nodes())} nodes)")

            t_group_start = time.time()
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            ctxs = list(sub_stage.run())
            group_wall_times[i] = time.time() - t_group_start
            assert len(ctxs) == 1, f"Expected 1 context from inner pipeline, got {len(ctxs)}"
            ctx = ctxs[0]

            allocation = ctx.get("allocation")
            chosen = {
                edge_tensor(edge): placement
                for edge in sub_workload.get_in_edges()
                if (placement := allocation.mapping.placement_of(edge)) is not None
            }
            self.sub_mappings = with_held_placements(sub_workloads, self.sub_mappings, chosen)
            group_latency = allocation.estimated_cycles
            total_latency += group_latency
            group_latencies[i] = group_latency
            group_columns[i] = _compute_columns(allocation)
            group_cycles[i] = allocation.estimated_cycles
            group_allocations[i] = AllocationIR.from_internal(allocation).model_dump()
            # Capture the memory-access breakdown (per core, per tensor, off-chip vs on-chip) so the
            # exploration result can show where the traffic goes -- the memory-wall view.
            mem_accesses = ctx.get("memory_accesses")
            try:
                group_memory_accesses[i] = mem_accesses.to_ir() if mem_accesses is not None else None
            except Exception as exc:  # noqa: BLE001 -- observability must never fail the solve
                logger.warning(f"Group {i}: could not serialise memory accesses: {exc}")
                group_memory_accesses[i] = None
            logger.info(f"Group {i} latency: {group_latency}")
            logger.info(f"Group {i} wall time: {group_wall_times[i]:.2f}s")
            final_ctx = ctx

        assert final_ctx is not None, "No groups processed"
        roofline = guarded("roofline", self._roofline, group_allocations, len(sub_workloads))
        final_ctx.set(
            roofline=roofline,
            total_latency=total_latency,
            group_latencies=group_latencies,
            group_columns=group_columns,
            group_cycles=group_cycles,
            group_wall_times=group_wall_times,
            group_allocations=group_allocations,
            group_memory_accesses=group_memory_accesses,
        )
        logger.info(f"Total latency across all groups: {total_latency}")
        logger.info(f"Per-group latencies: {group_latencies}")
        yield final_ctx

    def _held_beside(self, sub_workloads: "Sequence[Workload]", i: int) -> "dict[Memory, dict[Tensor, float]]":
        """What the off-chip memories hold beside the inputs group ``i`` places (:func:`offchip_bits`), the inputs later
        groups place a block in each; nothing to hold beside with one off-chip memory."""
        if len(self.accelerator.offchip_memories) <= 1:
            return {}
        mapping = self.sub_mappings[i]
        placing = {
            edge_tensor(edge)
            for edge in sub_workloads[i].get_in_edges()
            if placement_options(self.accelerator, mapping, edge)
        }
        held = offchip_bits(self.accelerator, sub_workloads, self.sub_mappings)
        return {memory: {t: b for t, b in tensors.items() if t not in placing} for memory, tensors in held.items()}

    def _roofline(self, group_allocations: dict[int, dict | None], n_groups: int) -> dict:
        """The run's roofline, from the point of each group that has one."""
        points = {i: ir["roofline"] for i, ir in group_allocations.items() if ir and ir.get("roofline")}
        return RooflineIR(**run_roofline(hardware_roofline(self.accelerator), points, n_groups)).model_dump()
