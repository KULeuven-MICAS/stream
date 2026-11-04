"""Generation stage: attention split along its sequence where it is placed so or its scores do not fit on chip."""

import logging
from collections.abc import Generator
from math import lcm
from typing import Any

from stream.hardware.architecture.accelerator import Accelerator
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.mapping.mapping import Mapping, check_held_capacity, viewed_placements
from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_parser import MappingParser
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.rewrites.flash_decoding import Attention, FlashDecodingRewrite
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

REWRITE = FlashDecodingRewrite()


def _part_counts(length: int, step: int) -> list[int]:
    """The part counts above one that split ``length`` evenly, ``step`` times a power of two, in increasing order."""
    counts, count = [], max(step, 2)
    while length % count == 0:
        counts.append(count)
        count *= 2
    return counts


def _check_input_capacity(accelerator: Accelerator, workload: Workload, runtime_args: dict) -> None:
    """Refuse workload inputs whose placements overflow an off-chip memory, before any mapping is generated."""
    inputs = Workload(list(workload.get_in_edges()))
    placed = {edge.name: runtime_args[edge.name] for edge in inputs.get_in_edges() if edge.name in runtime_args}
    placements = MappingFactory({"runtime_args": placed}, inputs, accelerator).create_edge_placements()
    check_held_capacity(accelerator, [inputs], [Mapping(edge_placements=placements)])


class FlashDecodingStage(Stage):
    """Split each attention's sequence into the fewest parts that fit, as ``attention_parts`` sets instead."""

    reads = ("accelerator", "workload", "output_path")
    optional_reads = ("mapping_path", "attention_parts", "nb_cols_to_use")
    writes = ("workload", "runtime_args")

    def run(self) -> Generator[StageContext]:
        workload: Workload = self.ctx.get("workload")
        parts = self.ctx.get("attention_parts", None)
        if parts is not None and (not isinstance(parts, int) or parts < 1):
            raise ValueError(f"attention_parts must be a positive integer, got {parts!r}")
        declared = None
        if (path := self.ctx.get("mapping_path", None)) is not None:
            declared = MappingParser(path, workload, self.ctx.get("accelerator")).parse_mapping_data()["runtime_args"]
            _check_input_capacity(self.ctx.get("accelerator"), workload, declared)
        for attention in REWRITE.find(workload):
            runtime_args = viewed_placements(workload, declared or {})
            splits = [
                (edge.name, axis, len(placement["memories"]))
                for edge, axis in attention.views(workload).items()
                if (placement := runtime_args.get(edge.name, {}).get("placement", {})).get("split") == axis
            ]
            count = parts or self._parts(workload, attention, declared or {}, lcm(*(n for *_, n in splits)))
            if count > 1:
                for name, axis, memories in splits:
                    if count % memories:
                        raise ValueError(
                            f"{name} is split along its axis {axis} over {memories} memories, which {count} parts of "
                            f"{attention.context.name}'s sequence do not split evenly"
                        )
                workload = REWRITE.apply(workload, attention, count)
                logger.info("Attention %s split into %d parts of its sequence", attention.context.name, count)
        runtime_args = viewed_placements(workload, declared) if declared is not None else None
        self.ctx.set(workload=workload, runtime_args=runtime_args)
        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _parts(self, workload: Workload, attention: Attention, declared: dict[str, Any], step: int) -> int:
        """One where no placement splits the sequence and the cut attention fits, else the fewest parts that fit, each
        a multiple of ``step``, by the runtime args the mapping file ``declared``."""
        names = {node.name for node in attention.nodes}
        if attention.length % step or (step == 1 and self._generator(workload, declared).fits_grouped(names)):
            return 1
        counts = _part_counts(attention.length, step)

        def fits(count: int, scale_tiles: bool) -> bool:
            rewritten = REWRITE.apply(workload, attention, count)
            return self._generator(rewritten, declared).fits_fused({f"{name}_part" for name in names}, scale_tiles)

        # Fitting is not monotone in the count (the tiler may fit 32 parts and not 256): try them in order.
        for scale_tiles in (False, True):
            if counts and fits(counts[-1], scale_tiles):
                return next(count for count in counts if count == counts[-1] or fits(count, scale_tiles))
        return step

    def _generator(self, workload: Workload, declared: dict[str, Any]) -> GenericMappingGenerator:
        runtime_args = viewed_placements(workload, declared) or None
        return GenericMappingGenerator(
            self.ctx.get("accelerator"),
            workload,
            self.ctx.get("output_path"),
            runtime_args=runtime_args,
            nb_cols_to_use=self.ctx.get("nb_cols_to_use", None),
        )


_: StageCallable = FlashDecodingStage
