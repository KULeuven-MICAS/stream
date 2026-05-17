import logging
from typing import Any

from zigzag.utils import open_yaml

from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import determine_fusion_cut_points

logger = logging.getLogger(__name__)


class FixedMappingGenerationStage(Stage):
    """Parse a fixed mapping YAML and build per-group Mappings in-memory.

    Reads the mapping YAML once, splits the workload into fusion groups,
    then builds a scoped Mapping per group via MappingFactory — no intermediate
    YAML files are written or read.

    Reads: accelerator, workload, mapping_path
    Writes: sub_workloads (list[Workload]), sub_mappings (list[Mapping])
    """

    REQUIRED_FIELDS = ("accelerator", "workload", "mapping_path")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.mapping_path: str = self.ctx.require_value("mapping_path", self.__class__.__name__)

    def run(self):
        mapping_data = self._parse_and_validate_yaml()
        cut_points = determine_fusion_cut_points(self.workload)
        sub_workloads = self.workload.split_fusion_groups(cut_points=cut_points)

        fused_groups_data = mapping_data["fused_groups"]
        assert len(fused_groups_data) == len(sub_workloads), (
            f"Fixed mapping has {len(fused_groups_data)} fused groups "
            f"but workload splits into {len(sub_workloads)} groups"
        )

        sub_mappings = []
        for i, sub_workload in enumerate(sub_workloads):
            per_group_data: dict[str, Any] = {
                "layers": mapping_data["layers"],
                "fused_groups": [fused_groups_data[i]],
                "runtime_args": mapping_data.get("runtime_args", {}),
            }
            factory = MappingFactory(per_group_data, sub_workload, self.accelerator)
            sub_mappings.append(factory.create())

        logger.info(f"Built {len(sub_mappings)} in-memory mappings from fixed YAML: {self.mapping_path}")
        self.ctx.set(sub_workloads=sub_workloads, sub_mappings=sub_mappings)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _parse_and_validate_yaml(self) -> dict[str, Any]:
        raw_data = open_yaml(self.mapping_path)
        validator = MappingValidator(raw_data)
        if not validator.validate():
            raise ValueError(f"Fixed mapping validation failed: {validator.errors}")
        return validator.normalized_data
