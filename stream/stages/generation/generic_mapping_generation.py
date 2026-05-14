import logging

from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class GenericMappingGenerationStage(Stage):
    """Generate per-fusion-group mapping YAMLs from workload+accelerator.

    Reads: accelerator, workload, output_path
    Writes: group_mapping_paths (list[str] of YAML file paths), sub_workloads (list[Workload])
    Delegates to: FusionGroupIterationStage (next in list_of_callables)
    """

    REQUIRED_FIELDS = ("accelerator", "workload", "output_path")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.output_path = self.ctx.require_value("output_path", self.__class__.__name__)

    def run(self):
        from stream.workload.workload import determine_fusion_cut_points  # noqa: PLC0415

        cut_points = determine_fusion_cut_points(self.workload)
        logger.info(f"Determined {len(cut_points)} fusion cut points: {cut_points}")

        generator = GenericMappingGenerator(
            accelerator=self.accelerator,
            workload=self.workload,
            output_dir=self.output_path,
        )
        group_mapping_paths, sub_workloads = generator.generate_all_groups(cut_points=cut_points)
        logger.info(f"Generated {len(group_mapping_paths)} group mapping(s): {group_mapping_paths}")

        # Write both paths AND sub_workloads to context so FusionGroupIterationStage
        # does NOT need to re-call split_fusion_groups() (avoids duplicate work per WARNING 2)
        self.ctx.set(group_mapping_paths=group_mapping_paths, sub_workloads=sub_workloads)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
