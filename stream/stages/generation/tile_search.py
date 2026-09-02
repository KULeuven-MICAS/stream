import logging
import os
from dataclasses import replace

from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.stages.generation.mapping_generation import save_infeasibility_report
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

GROWTH_FACTORS = (2, 4)


class TileSearchStage(Stage):
    """Let the optimizer choose each fused group's intra-core tile size.

    The mapping's declared tiling is the seed: the finest granule the compiled kernels
    accept. Multiples of the seed are valid a priori -- they keep kernel-call granularity
    and still divide the same extents -- so the candidate set needs no external pruning
    knowledge. Every candidate is priced by the same tiling + cost + allocation tail that
    prices everything else, so feasibility (memory, fifo depth, DMA channels) and worth
    (latency under the calibrated kernel costs) come from one model, and the best-latency
    solve wins. Off unless the context carries ``tile_search=True``.
    """

    REQUIRED_FIELDS = ("workload", "mapping", "output_path")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.output_path: str = self.ctx.get("output_path")
        self.enabled: bool = bool(self.ctx.get("tile_search", False))

    def _candidates(self) -> list[Mapping]:
        seeds = [self.mapping]
        groups = self.mapping.fused_groups
        for gi, group in enumerate(groups):
            for ti, (dim, tile) in enumerate(group.intra_core_tiling):
                extent = self.workload.get_dimension_size(dim)
                for factor in GROWTH_FACTORS:
                    grown = tile * factor
                    if grown > extent or extent % grown != 0:
                        continue
                    tiling = list(group.intra_core_tiling)
                    tiling[ti] = (dim, grown)
                    new_groups = list(groups)
                    new_groups[gi] = replace(group, intra_core_tiling=tuple(tiling))
                    seeds.append(self.mapping.with_fused_groups(new_groups))
        return seeds

    def run(self):
        candidates = self._candidates() if self.enabled else [self.mapping]
        if len(candidates) == 1:
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            yield from sub_stage.run()
            return

        base = dict(self.ctx.data)
        best_context = None
        best_latency = float("inf")
        best_index = None
        for i, mapping in enumerate(candidates):
            tiling = {str(d): t for g in mapping.fused_groups for d, t in g.intra_core_tiling}
            candidate_path = os.path.join(self.output_path, f"tile_{i}")
            os.makedirs(candidate_path, exist_ok=True)
            self.ctx.data = dict(base)
            self.ctx.set(mapping=mapping, output_path=candidate_path)
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            try:
                ctxs = list(sub_stage.run())
                assert len(ctxs) == 1, f"Expected exactly one context, but got {len(ctxs)}"
                latency = ctxs[0].get("scheduler").latency_total
            except InfeasibleAllocationError as e:
                save_infeasibility_report(candidate_path, e.report)
                logger.info("Tile candidate %s is infeasible: %s", tiling, e.report.summary)
                continue
            except (RuntimeError, ValueError, AssertionError) as e:
                logger.info("Tile candidate %s failed: %s", tiling, e)
                continue
            logger.info("Tile candidate %s: latency %s", tiling, latency)
            if latency < best_latency:
                best_latency = latency
                best_index = i
                best_context = StageContext(data=dict(ctxs[0].data))
        if best_context is None:
            raise RuntimeError("No feasible tile candidate; the seed tiling itself did not allocate.")
        logger.info("Tile search chose candidate %d of %d (latency %s)", best_index, len(candidates), best_latency)
        # Downstream consumers (codegen, hosts reading the output tree) expect the group's own path.
        best_context.set(output_path=self.output_path)
        self.ctx = best_context
        yield best_context
