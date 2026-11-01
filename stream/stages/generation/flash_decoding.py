"""Generation stage: attention split along its sequence where it is placed so or its scores do not fit on chip."""

import logging
from collections.abc import Generator
from math import lcm
from typing import Any

from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.generation.generic_mapping_generation import runtime_args_of
from stream.stages.stage import Stage, StageCallable
from stream.workload.rewrites import get_workload_rewrite
from stream.workload.rewrites.flash_decoding import Attention
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


def _part_counts(length: int) -> list[int]:
    """The part counts above one that split ``length`` evenly, powers of two first, each in increasing order."""
    divisors = [n for n in range(2, length + 1) if length % n == 0]
    return sorted(divisors, key=lambda n: (n & (n - 1) != 0, n))


def _viewed_placements(
    workload: Workload, attention: Attention, parts: int, runtime_args: dict[str, Any]
) -> dict[str, Any]:
    """``runtime_args`` with each viewed input's placement split counted in its new axes, the sequence's as the
    parts'."""
    viewed = dict(runtime_args)
    for edge, axis in attention.views(workload).items():
        placement = runtime_args.get(edge.name, {}).get("placement", {})
        if (split := placement.get("split")) is None:
            continue
        if split == axis and parts % len(placement["memories"]):
            raise ValueError(
                f"{edge.name} is split along its axis {axis} over {len(placement['memories'])} memories, which "
                f"{parts} parts of {attention.context.name}'s sequence do not split evenly"
            )
        viewed[edge.name] = runtime_args[edge.name] | {"placement": placement | {"split": split + (split > axis)}}
    return viewed


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
        runtime_args = runtime_args_of(self.ctx)
        for attention in rewrite.find(workload):
            count = parts or self._parts(workload, attention, runtime_args or {})
            if count > 1:
                if runtime_args is not None:
                    runtime_args = _viewed_placements(workload, attention, count, runtime_args)
                workload = rewrite.apply(workload, attention, count)
                logger.info("Attention %s split into %d parts of its sequence", attention.context.name, count)
        self.ctx.set(workload=workload, runtime_args=runtime_args)
        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _parts(self, workload: Workload, attention: Attention, runtime_args: dict[str, Any]) -> int:
        """The fewest parts, a multiple of the memories the sequence is split over, that fit, else that multiple."""
        edges = {edge.outputs[0]: edge.name for edge in workload.get_in_edges()}
        placed = [
            len(placement["memories"])
            for tensor, axis in attention.sequence_reads()
            if (placement := runtime_args.get(edges.get(tensor, ""), {}).get("placement", {})).get("split") == axis
        ]
        step = lcm(*placed) if placed else 1
        names = {node.name for node in attention.nodes}
        if attention.length % step or (step == 1 and self._fits(workload, names, runtime_args)):
            return 1
        rewrite = get_workload_rewrite("flash_decoding")
        for count in (n for n in _part_counts(attention.length) if n % step == 0):
            viewed = _viewed_placements(workload, attention, count, runtime_args)
            if self._fits(rewrite.apply(workload, attention, count), {f"{name}_part" for name in names}, viewed):
                return count
        return step

    def _fits(self, workload: Workload, names: set[str], runtime_args: dict[str, Any]) -> bool:
        generator = GenericMappingGenerator(
            self.ctx.get("accelerator"), workload, self.ctx.get("output_path"), runtime_args=runtime_args or None
        )
        return generator.fits_fused(names)


_: StageCallable = FlashDecodingStage
