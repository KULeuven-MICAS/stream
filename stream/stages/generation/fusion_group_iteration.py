import logging
import os
import time

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class FusionGroupIterationStage(Stage):
    """Iterate over fusion groups, running the inner pipeline once per group.

    Supports two modes:
    - In-memory: sub_mappings provided → sets mapping directly per group (no MappingParserStage needed)
    - File-based: group_mapping_paths provided → sets mapping_path per group (MappingParserStage reads it)

    Reads: sub_workloads, output_path, and either sub_mappings or group_mapping_paths
    Writes: total_latency (float), group_latencies (dict), group_wall_times (dict)
    """

    REQUIRED_FIELDS = ("accelerator", "workload", "output_path", "sub_workloads")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.output_path = self.ctx.require_value("output_path", self.__class__.__name__)
        self.sub_workloads = self.ctx.require_value("sub_workloads", self.__class__.__name__)
        self.sub_mappings = self.ctx.get("sub_mappings")
        self.group_mapping_paths = self.ctx.get("group_mapping_paths")
        if self.sub_mappings is None and self.group_mapping_paths is None:
            raise ValueError(
                f"{self.__class__.__name__} requires either 'sub_mappings' or 'group_mapping_paths' in context"
            )

    def run(self):
        sub_workloads = self.sub_workloads
        total_latency = 0.0
        group_latencies: dict[int, float] = {}
        group_wall_times: dict[int, float] = {}
        final_ctx = None

        if self.sub_mappings is not None:
            assert len(sub_workloads) == len(self.sub_mappings), (
                f"Mismatch: {len(sub_workloads)} sub-workloads vs {len(self.sub_mappings)} sub-mappings"
            )
        else:
            assert len(sub_workloads) == len(self.group_mapping_paths), (
                f"Mismatch: {len(sub_workloads)} sub-workloads vs {len(self.group_mapping_paths)} mapping paths"
            )

        for i, sub_workload in enumerate(sub_workloads):
            group_output = os.path.join(self.output_path, f"group_{i}")
            os.makedirs(group_output, exist_ok=True)

            ctx_updates = dict(workload=sub_workload, output_path=group_output)
            if self.sub_mappings is not None:
                ctx_updates["mapping"] = self.sub_mappings[i]
            else:
                ctx_updates["mapping_path"] = self.group_mapping_paths[i]
            self.ctx.set(**ctx_updates)

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
