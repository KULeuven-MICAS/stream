import logging
import os
import time

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class FusionGroupIterationStage(Stage):
    """Iterate over fusion groups, running the inner pipeline once per group.

    Per D-01/D-02: Wraps MappingParserStage through MemoryAccessesEstimationStage.
    Per D-04: If split_fusion_groups() returns a single Workload (no FusionEdge nodes),
    runs inner stages once with the original workload.
    Per D-05: Aggregation = total latency = sum of per-group latencies.

    Reads: sub_workloads, group_mapping_paths, output_path
    Writes: total_latency (float)
    Inner pipeline (via list_of_callables): MappingParserStage -> TilingGenerationStage ->
        CoreCostEstimationStage -> ConstraintOptimizationAllocationStage -> MemoryAccessesEstimationStage
    """

    REQUIRED_FIELDS = ("accelerator", "workload", "output_path", "group_mapping_paths", "sub_workloads")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.output_path = self.ctx.require_value("output_path", self.__class__.__name__)
        self.group_mapping_paths = self.ctx.require_value("group_mapping_paths", self.__class__.__name__)
        self.sub_workloads = self.ctx.require_value("sub_workloads", self.__class__.__name__)

    def run(self):
        sub_workloads = self.sub_workloads
        group_mapping_paths = self.group_mapping_paths
        total_latency = 0.0
        group_latencies: dict[int, float] = {}
        group_wall_times: dict[int, float] = {}
        final_ctx = None

        assert len(sub_workloads) == len(group_mapping_paths), (
            f"Mismatch: {len(sub_workloads)} sub-workloads vs {len(group_mapping_paths)} mapping paths"
        )

        for i, (sub_workload, mapping_path) in enumerate(zip(sub_workloads, group_mapping_paths, strict=False)):
            group_output = os.path.join(self.output_path, f"group_{i}")
            os.makedirs(group_output, exist_ok=True)

            # Per D-02/Pitfall 5: Set per-group context before each inner run
            self.ctx.set(
                workload=sub_workload,
                mapping_path=mapping_path,
                output_path=group_output,
            )

            logger.info(f"Running inner pipeline for group {i} ({len(sub_workload.get_computation_nodes())} nodes)")

            t_group_start = time.time()
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            ctxs = list(sub_stage.run())
            group_wall_times[i] = time.time() - t_group_start
            assert len(ctxs) == 1, f"Expected 1 context from inner pipeline, got {len(ctxs)}"
            ctx = ctxs[0]

            scheduler = ctx.get("scheduler")
            group_latency = scheduler.latency_total
            total_latency += group_latency
            group_latencies[i] = group_latency
            logger.info(f"Group {i} latency: {group_latency}")
            logger.info(f"Group {i} wall time: {group_wall_times[i]:.2f}s")
            final_ctx = ctx

        # Per D-05: Store aggregated result
        assert final_ctx is not None, "No groups processed"
        final_ctx.set(total_latency=total_latency, group_latencies=group_latencies, group_wall_times=group_wall_times)
        logger.info(f"Total latency across all groups: {total_latency}")
        logger.info(f"Per-group latencies: {group_latencies}")
        yield final_ctx
