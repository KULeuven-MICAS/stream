"""Generation stage: attention split along its sequence where it is placed so or its scores do not fit on chip."""

import logging
from collections.abc import Generator
from math import lcm
from typing import Any

from stream.mapping.generic_generator import GenericMappingGenerator
from stream.mapping.mapping import viewed_placements
from stream.stages.context import StageContext
from stream.stages.generation.generic_mapping_generation import check_input_capacity, runtime_args_of
from stream.stages.stage import Stage, StageCallable
from stream.workload.rewrites import get_workload_rewrite
from stream.workload.rewrites.flash_decoding import Attention
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


def _part_counts(length: int, step: int) -> list[int]:
    """The part counts above one that split ``length`` evenly, ``step`` times a power of two, in increasing order."""
    counts, count = [], max(step, 2)
    while length % count == 0:
        counts.append(count)
        count *= 2
    return counts


def _check_parts(workload: Workload, attention: Attention, parts: int, runtime_args: dict[str, Any]) -> None:
    """Refuse ``parts`` where they do not split evenly over the memories of a placement splitting the sequence."""
    for edge, axis in attention.views(workload).items():
        placement = runtime_args.get(edge.name, {}).get("placement", {})
        if placement.get("split") == axis and parts % len(placement["memories"]):
            raise ValueError(
                f"{edge.name} is split along its axis {axis} over {len(placement['memories'])} memories, which "
                f"{parts} parts of {attention.context.name}'s sequence do not split evenly"
            )


class FlashDecodingStage(Stage):
    """Split each attention's sequence into the fewest parts that fit, as ``attention_parts`` sets instead."""

    reads = ("accelerator", "workload", "output_path")
    optional_reads = ("mapping_path", "attention_parts")
    writes = ("workload", "runtime_args")

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.get("workload")
        parts = self.ctx.get("attention_parts", None)
        if parts is not None and (not isinstance(parts, int) or parts < 1):
            raise ValueError(f"attention_parts must be a positive integer, got {parts!r}")
        rewrite = get_workload_rewrite("flash_decoding")
        declared = runtime_args_of(self.ctx)
        if declared is not None:
            check_input_capacity(self.ctx.get("accelerator"), workload, declared)
        for attention in rewrite.find(workload):
            count = parts or self._parts(workload, attention, declared or {})
            if count > 1:
                if declared is not None:
                    _check_parts(workload, attention, count, viewed_placements(workload, declared))
                workload = rewrite.apply(workload, attention, count)
                logger.info("Attention %s split into %d parts of its sequence", attention.context.name, count)
        runtime_args = viewed_placements(workload, declared) if declared is not None else None
        self.ctx.set(workload=workload, runtime_args=runtime_args)
        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _parts(self, workload: Workload, attention: Attention, declared: dict[str, Any]) -> int:
        """One where no placement splits the sequence and the cut attention fits, else the fewest parts that fit, by
        the runtime args the mapping file ``declared``."""
        runtime_args = viewed_placements(workload, declared)
        edges = {edge.outputs[0]: edge.name for edge in workload.get_in_edges()}
        placed = [
            len(placement["memories"])
            for tensor, axis in attention.sequence_reads()
            if (placement := runtime_args.get(edges.get(tensor, ""), {}).get("placement", {})).get("split") == axis
        ]
        step = lcm(*placed) if placed else 1
        names = {node.name for node in attention.nodes}
        if attention.length % step or (step == 1 and self._fits_cut(workload, names, runtime_args)):
            return 1
        counts = _part_counts(attention.length, step)
        rewrite = get_workload_rewrite("flash_decoding")

        def fits(count: int, scale_tiles: bool) -> bool:
            rewritten = rewrite.apply(workload, attention, count)
            viewed = viewed_placements(rewritten, declared)
            return self._fits(rewritten, {f"{name}_part" for name in names}, viewed, scale_tiles)

        # Fitting is not monotone in the count (the tiler may fit 32 parts and not 256): try them in order.
        for scale_tiles in (False, True):
            if counts and fits(counts[-1], scale_tiles):
                return next(count for count in counts if count == counts[-1] or fits(count, scale_tiles))
        return step

    def _generator(self, workload: Workload, runtime_args: dict[str, Any]) -> GenericMappingGenerator:
        return GenericMappingGenerator(
            self.ctx.get("accelerator"), workload, self.ctx.get("output_path"), runtime_args=runtime_args or None
        )

    def _fits(self, workload: Workload, names: set[str], runtime_args: dict[str, Any], scale_tiles: bool) -> bool:
        return self._generator(workload, runtime_args).fits_fused(names, scale_tiles)

    def _fits_cut(self, workload: Workload, names: set[str], runtime_args: dict[str, Any]) -> bool:
        """Whether the attention ``names`` fits, cut where the mapping generator would cut it."""
        generator = self._generator(workload, runtime_args)
        return generator.fits_grouped(generator.groups(), names)


_: StageCallable = FlashDecodingStage
