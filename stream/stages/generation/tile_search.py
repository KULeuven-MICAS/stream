import logging
import os
from dataclasses import replace

from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.generation.mapping_generation import save_infeasibility_report
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

GROWTH_FACTORS = (2, 4)


class TileSearchStage(Stage):
    """Let the optimizer choose each fused group's intra-core tile size.

    The mapping's declared tiling is the seed: the finest granule the compiled kernels
    accept. Growth is offered only along the group's growable dims -- the ones every
    declaring kernel consumes in a run-time loop -- since a compiled block's own
    dimensions never see more than one granule per call. Every candidate is priced by the same tiling + cost + allocation tail that
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

    def _candidates(self) -> list[tuple[object, Mapping]]:
        seeds: list[tuple[object, Mapping]] = [(None, self.mapping)]
        groups = self.mapping.fused_groups
        for gi, group in enumerate(groups):
            for ti, (dim, tile) in enumerate(group.intra_core_tiling):
                if dim not in group.growable_dims:
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
            self.ctx.data.pop("placement_alternatives", None)
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            yield from sub_stage.run()
            return
        # Placement candidates share no structure, so a selector-gated union model
        # decomposes exactly into one solve per candidate; every (placement, tile)
        # pair is priced by the same tail and the best latency deploys.
        attempts = [self.mapping, *(self.ctx.data.pop("placement_alternatives", None) or [])]
        # Reserves are measured-refuted shapes: they never rival on price, they only
        # stand in when no priced placement allocates.
        reserves = self.ctx.data.pop("placement_reserves", None) or []
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
        yield self._finish(*best, None)

    def _search(self):
        candidates = self._candidates()
        base = dict(self.ctx.data)
        best_context = None
        best_latency = float("inf")
        best_index = None
        seed_error: Exception | None = None
        seed_latency = float("inf")
        # Growth along one dimension only adds memory pressure and per-iteration latency,
        # so once a grown tile loses to the seed, larger growths of the same dimension
        # are skipped rather than solved.
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
                logger.info("Tile candidate %s is infeasible: %s", tiling, e.report.summary)
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
                if self._seed_exhausted(ctxs[0]) or os.environ.get("STREAM_TILE_FORCE") == "seed":
                    best_latency, best_index = latency, 0
                    best_context = StageContext(data=dict(ctxs[0].data))
                    break
            elif latency >= seed_latency:
                dead_dims.add(grown_dim)
            if os.environ.get("STREAM_TILE_FORCE") == "largest":
                # Calibration probe: deploy the largest feasible tile; "seed" above stops
                # at the granule seed, so the pair brackets what the search itself buys.
                best_latency, best_index = latency, i
                best_context = StageContext(data=dict(ctxs[0].data))
                continue
            if latency < best_latency:
                best_latency = latency
                best_index = i
                best_context = StageContext(data=dict(ctxs[0].data))
        if best_context is None:
            # The seed is the caller's own finest granule: its failure is the real error.
            raise RuntimeError("No feasible tile candidate.") from seed_error
        return best_context, best_index, best_latency, len(candidates)

    def _finish(self, best_context, best_index, best_latency, n, seed_error):
        if best_context is None:
            # The seed is the caller's own declared tiling: its failure is the real error,
            # not a search outcome.
            raise RuntimeError("No feasible tile candidate.") from seed_error
        logger.info("Tile search chose candidate %d of %d (latency %s)", best_index, n, best_latency)
        # Downstream consumers (codegen, hosts reading the output tree) expect the group's own path.
        best_context.set(output_path=self.output_path)
        self.ctx = best_context
        return best_context

    def _seed_exhausted(self, ctx) -> bool:
        """A seed that alone hits the solve budget is not a model to search over."""
        stats = ctx.get("scheduler").solve_stats
        if stats is not None and stats.status == "TIME_LIMIT":
            logger.info("Seed solve hit the time limit; skipping tile candidates")
            return True
        return False

    def _evaluate(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        ctxs = list(sub_stage.run())
        assert len(ctxs) == 1, f"Expected exactly one context, but got {len(ctxs)}"
        return ctxs, ctxs[0].get("scheduler").latency_total
