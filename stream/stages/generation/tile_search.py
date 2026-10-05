import logging
import os
from dataclasses import replace

from stream.ir.infeasibility import InfeasibleAllocationError, save_infeasibility_report
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

GROWTH_FACTORS = (2, 4)


class TileSearchStage(Stage):
    """Let the optimizer choose each fused group's intra-core tile size."""

    reads = ("workload", "mapping", "output_path")
    optional_reads = ("tile_search",)
    writes = ("mapping", "output_path", "placement_alternatives", "placement_reserves")
    result_reads = ("allocation",)
    result_writes = ("output_path",)

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.output_path: str = self.ctx.get("output_path")
        self.enabled: bool = bool(self.ctx.get("tile_search", False))

    def _candidates(self) -> list[tuple[object, Mapping]]:
        seeds: list[tuple[object, Mapping]] = [(None, self.mapping)]
        groups = self.mapping.fused_groups
        for gi, group in enumerate(groups):
            for ti, (dim, tile) in enumerate(group.intra_core_tiling):
                if dim not in group.runtime_dims:
                    continue
                extent = self.workload.get_dimension_size(dim)
                for factor in GROWTH_FACTORS:
                    grown = tile * factor
                    if grown > extent or extent % grown != 0:
                        continue
                    tiling = list(group.intra_core_tiling)
                    tiling[ti] = (dim, grown)
                    new_groups = list(groups)
                    new_groups[gi] = replace(group, intra_core_tiling=tuple(tiling))
                    seeds.append(((gi, ti), self.mapping.with_fused_groups(new_groups)))
        return seeds

    def run(self):
        if not self.enabled:
            self.ctx.pop("placement_alternatives")
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            yield from sub_stage.run()
            return
        attempts = [self.mapping, *(self.ctx.pop("placement_alternatives") or [])]
        reserves = self.ctx.pop("placement_reserves") or []
        entry = dict(self.ctx.data)
        best = None
        error: Exception | None = None
        for a, mapping in enumerate(attempts + reserves):
            if best is not None and a >= len(attempts):
                break
            self._attempt = a
            self.ctx.data = dict(entry)
            self.ctx.set(mapping=mapping)
            self.mapping = mapping
            try:
                found = self._search()
            except RuntimeError as e:
                error = error or e
                logger.info("Placement %d has no feasible tile", a)
                continue
            logger.info("Placement %d prices at %s", a, found[2])
            if best is None or found[2] < best[2]:
                best = found
                if a >= len(attempts):
                    break
        if best is None:
            raise error or RuntimeError("No feasible placement.")
        yield self._finish(*best)

    def _search(self):
        candidates = self._candidates()
        base = dict(self.ctx.data)
        best_context = None
        best_latency = float("inf")
        best_index = None
        seed_error: Exception | None = None
        seed_latency, timed_out = float("inf"), False
        dead_dims: set[object] = set()
        for i, (grown_dim, mapping) in enumerate(candidates):
            if grown_dim in dead_dims:
                continue
            tiling = {str(d): t for g in mapping.fused_groups for d, t in g.intra_core_tiling}
            prefix = f"p{self._attempt}_" if self._attempt else ""
            candidate_path = os.path.join(self.output_path, f"{prefix}tile_{i}")
            os.makedirs(candidate_path, exist_ok=True)
            self.ctx.data = dict(base)
            self.ctx.set(mapping=mapping, output_path=candidate_path)
            try:
                ctxs, latency = self._evaluate()
            except InfeasibleAllocationError as e:
                save_infeasibility_report(candidate_path, e.report)
                timed_out |= e.report.status == "TIME_LIMIT"
                logger.info("Tile candidate %s is unpriced: %s", tiling, e.report.summary)
                if i == 0:
                    seed_error = e
                dead_dims.add(grown_dim)
                continue
            except (RuntimeError, ValueError, AssertionError) as e:
                if i == 0:
                    seed_error = e
                logger.info("Tile candidate %s failed: %s", tiling, e)
                dead_dims.add(grown_dim)
                continue
            logger.info("Tile candidate %s: latency %s", tiling, latency)
            if i == 0:
                seed_latency = latency
                if self._seed_exhausted(ctxs[0]):
                    best_latency, best_index = latency, 0
                    best_context = StageContext(data=dict(ctxs[0].data))
                    break
            elif latency >= seed_latency:
                dead_dims.add(grown_dim)
            if latency < best_latency:
                best_latency = latency
                best_index = i
                best_context = StageContext(data=dict(ctxs[0].data))
        if best_context is None:
            reason = "the allocation solve ran out of time" if timed_out else "none is feasible"
            raise RuntimeError(f"No tile candidate was priced: {reason}.") from seed_error
        return best_context, best_index, best_latency, len(candidates)

    def _finish(self, best_context, best_index, best_latency, n):
        logger.info("Tile search chose candidate %d of %d (latency %s)", best_index, n, best_latency)
        best_context.set(output_path=self.output_path)
        self.ctx = best_context
        return best_context

    def _seed_exhausted(self, ctx) -> bool:
        """A seed that alone hits the solve budget is not a model to search over."""
        if ctx.get("allocation").solution.solve_stats.status == "TIME_LIMIT":
            logger.info("Seed solve hit the time limit; skipping tile candidates")
            return True
        return False

    def _evaluate(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        ctxs = list(sub_stage.run())
        assert len(ctxs) == 1, f"Expected exactly one context, but got {len(ctxs)}"
        return ctxs, ctxs[0].get("allocation").cost_to_rank
