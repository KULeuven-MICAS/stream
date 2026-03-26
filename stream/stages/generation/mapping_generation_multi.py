import concurrent.futures as cf
import copy
import logging
import os

import yaml

from stream.mapping.generator import MappingGenerator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


def _evaluate_one_mapping(  # top-level for clean pickling if you ever switch to processes
    *,
    i: int,
    variant,
    mapping: dict,
    output_path: str,
    workload,
    mapping_generator: MappingGenerator,
    list_of_callables: list[StageCallable],
    base_ctx: StageContext,
) -> tuple[int, float, str | None]:
    """
    Worker for a single mapping evaluation.
    Returns (i, latency, mapping_path) or (i, inf, None) if evaluation failed.
    Uses a per-task context copy to avoid shared-mutable-state races.
    """
    output_path_i = os.path.join(output_path, f"{i}")
    os.makedirs(output_path_i, exist_ok=True)

    # Save mapping into the same folder passed onward to substages
    mapping_path = mapping_generator.save_mapping(
        mapping=mapping,
        variant=variant,
        idx=i,
        output_dir=output_path_i,
    )

    # Important: never mutate the shared ctx, use a per-task copy
    ctx_i = copy.deepcopy(base_ctx)
    ctx_i.set(
        workload=workload,
        mapping_path=mapping_path,
        output_path=output_path_i,
    )

    logger.info(f"Evaluating mapping: {mapping_path}")
    sub_stage = list_of_callables[0](list_of_callables[1:], ctx_i)

    try:
        ctxs = list(sub_stage.run())
    except RuntimeError as e:
        logger.error(f"Error evaluating mapping {mapping_path}: {e}")
        return (i, float("inf"), None)

    if len(ctxs) != 1:
        logger.error(f"Expected exactly one context, but got {len(ctxs)} for {mapping_path}")
        return (i, float("inf"), None)

    ctx_out = ctxs[0]
    scheduler = ctx_out.get("scheduler", None)
    if scheduler is None:
        logger.error(f"No scheduler found in context for {mapping_path}")
        return (i, float("inf"), None)

    latency = float(scheduler.latency_total)

    # Save the latency to yaml for later analysis
    latency_yaml_path = os.path.join(output_path_i, "latency.yaml")
    with open(latency_yaml_path, "w") as f:
        yaml.safe_dump({"latency": latency}, f)

    logger.info(f"Mapping {mapping_path} has latency {latency}")
    return (i, latency, mapping_path)


class MappingGenerationMultiThreadedStage(Stage):
    REQUIRED_FIELDS = ("accelerator", "workload")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.output_path = self.ctx.require_value("output_path", self.__class__.__name__)
        seq_len_tile_size = self.ctx.get("seq_len_tile_size", 32)
        embedding_tile_size = self.ctx.get("embedding_tile_size", 128)
        hidden_tile_size = self.ctx.get("hidden_tile_size", 64)
        last_gemm_down = self.ctx.get("last_gemm_down", False)
        max_nb_mappings = self.ctx.get("max_nb_mappings", 200)

        # Thread pool knobs (optional in ctx)
        self.max_workers = int(self.ctx.get("max_workers", os.cpu_count() or 4))
        # how many tasks to keep queued at once (limits memory growth)
        self.max_in_flight = int(self.ctx.get("max_in_flight", self.max_workers))

        self.mapping_generator = MappingGenerator(
            accelerator=self.accelerator,
            workload=self.workload,
            output_dir=self.output_path,
            seq_len_tile_size=seq_len_tile_size,
            embedding_tile_size=embedding_tile_size,
            hidden_tile_size=hidden_tile_size,
            last_gemm_down=last_gemm_down,
            max_variants=max_nb_mappings,
            layer_core_splits={
                "Gemm_Left": [4, 8, 16],
                "Gemm_Right": [4, 8, 16],
                "Silu": [1, 4],
                "Elt_Mul": [1, 4],
                "Gemm_Down": [4, 8, 16],  # only used if last_gemm_down=True
            },
        )

    def run(self):
        best_mapping_path = None
        best_latency = float("inf")

        # Snapshot once; do NOT mutate self.ctx in parallel loop.
        base_ctx = copy.deepcopy(self.ctx)

        # Producer-consumer with backpressure:
        # submit up to max_in_flight tasks; as tasks finish, submit more.
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures: dict[cf.Future, int] = {}

            def submit_one(i, variant, mapping):
                fut = ex.submit(
                    _evaluate_one_mapping,
                    i=i,
                    variant=variant,
                    mapping=mapping,
                    output_path=self.output_path,
                    workload=self.workload,
                    mapping_generator=self.mapping_generator,
                    list_of_callables=self.list_of_callables,
                    base_ctx=base_ctx,
                )
                futures[fut] = i

            gen_iter = self.mapping_generator.run()

            # Prime the queue
            for _ in range(self.max_in_flight):
                try:
                    i, variant, mapping = next(gen_iter)
                except StopIteration:
                    break
                submit_one(i, variant, mapping)

            while futures:
                done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    _ = futures.pop(fut, None)

                    # Always refill one slot when a task completes, regardless of
                    # success or failure, so the pool never drains due to errors.
                    try:
                        i2, variant2, mapping2 = next(gen_iter)
                    except StopIteration:
                        pass
                    else:
                        submit_one(i2, variant2, mapping2)

                    try:
                        _, latency, mapping_path = fut.result()
                    except Exception as e:
                        logger.exception(f"Worker crashed: {e}")
                        continue

                    if mapping_path is None:
                        continue

                    if latency < best_latency:
                        best_latency = latency
                        best_mapping_path = mapping_path

        logger.info(f"Best mapping found with latency {best_latency}: {best_mapping_path}")

        # Re-run the best mapping once (single-thread) to produce the best_context to yield.
        # This avoids returning complex ctx objects from worker threads.
        if best_mapping_path is None:
            yield None
            return

        ctx_best = copy.deepcopy(self.ctx)
        ctx_best.set(
            workload=self.workload,
            mapping_path=best_mapping_path,
            output_path=os.path.dirname(best_mapping_path),
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], ctx_best)
        ctxs = list(sub_stage.run())
        assert len(ctxs) == 1, f"Expected exactly one context, but got {len(ctxs)}"
        yield ctxs[0]
