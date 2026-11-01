import logging

from stream.hardware.architecture.accelerator import Accelerator
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.mapping.mapping import Mapping, check_offchip_capacity
from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_parser import MappingParser
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.node import HasIterationSpace
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

#: Sentinel for ``fusion_cut_points``: give every layer its own fusion group.
PER_LAYER = "per-layer"


def runtime_args_of(ctx: StageContext) -> dict | None:
    """The runtime args placing the workload's edges: as a rewrite left them, else the mapping file's."""
    if (runtime_args := ctx.get("runtime_args", None)) is not None:
        return runtime_args
    if (path := ctx.get("mapping_path", None)) is None:
        return None
    return MappingParser(path, ctx.get("workload"), ctx.get("accelerator")).parse_mapping_data()["runtime_args"]


def check_input_capacity(accelerator: Accelerator, workload: Workload, runtime_args: dict) -> None:
    """Refuse workload inputs whose placements overflow an off-chip memory, before any mapping is generated."""
    inputs = Workload(list(workload.get_in_edges()))
    placed = {edge.name: runtime_args[edge.name] for edge in inputs.get_in_edges() if edge.name in runtime_args}
    placements = MappingFactory({"runtime_args": placed}, inputs, accelerator).create_edge_placements()
    check_offchip_capacity(accelerator, [inputs], [Mapping(edge_placements=placements)])


class GenericMappingGenerationStage(Stage):
    """Generate each fusion group's mapping from workload+accelerator, written as YAML beside the run."""

    reads = ("accelerator", "workload", "output_path")
    optional_reads = ("fusion_cut_points", "intra_core_tiling", "mapping_path", "runtime_args")
    writes = ("sub_workloads", "sub_mappings")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.get("accelerator")
        self.workload = self.ctx.get("workload")
        self.output_path = self.ctx.get("output_path")

    def run(self):
        # Caller-supplied cuts win over the ones the generator derives (the affine barriers, and where a group's
        # weights cannot stay on its cores).
        cut_points = self.ctx.get("fusion_cut_points", None)
        if cut_points == PER_LAYER:
            # A cut point ends its group, so cut after every layer but the last -> one group per layer.
            names = [n.name for n in self.workload.dataflow_sort() if isinstance(n, HasIterationSpace)]
            cut_points = names[:-1]
            logger.info(f"Layer-by-layer: cutting at every layer boundary ({len(cut_points)} cuts)")
        elif cut_points is None:
            logger.info("Fusion cut points derived by the mapping generator")
        else:
            logger.info(f"Using {len(cut_points)} caller-supplied fusion cut points: {cut_points}")

        runtime_args = runtime_args_of(self.ctx)
        if runtime_args is not None:
            check_input_capacity(self.accelerator, self.workload, runtime_args)
        generator = GenericMappingGenerator(
            accelerator=self.accelerator,
            workload=self.workload,
            output_dir=self.output_path,
            # Optional layer-fusion intra-core tiling (None -> generic mapper's trivial no-op tiling).
            intra_core_tiling=self.ctx.get("intra_core_tiling", None),
            runtime_args=runtime_args,
        )
        group_mapping_paths, sub_workloads = generator.generate_all_groups(cut_points=cut_points)
        logger.info(f"Generated {len(group_mapping_paths)} group mapping(s): {group_mapping_paths}")

        sub_mappings = [
            MappingParser(path, sub_workload, self.accelerator).run()
            for path, sub_workload in zip(group_mapping_paths, sub_workloads, strict=True)
        ]
        self.ctx.set(sub_workloads=sub_workloads, sub_mappings=sub_mappings)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
