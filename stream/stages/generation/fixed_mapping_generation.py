import logging
from typing import Any

from zigzag.utils import open_yaml

from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import InEdge, OutEdge, determine_fusion_cut_points

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
        fused_groups_data = mapping_data["fused_groups"]

        cut_points = determine_fusion_cut_points(self.workload)
        sub_workloads = self.workload.split_fusion_groups(cut_points=cut_points)

        # A fixed mapping is authoritative about its own grouping. If it declares
        # a different number of fused groups than the workload heuristic found
        # (e.g. a SwiGLU split into a gate/up/SiLU/mul group and a separate
        # down-projection group, which has no MaxPool/residual cut to detect),
        # derive the cut points from the mapping's group->layer assignment instead.
        if len(fused_groups_data) != len(sub_workloads):
            cut_points = self._cut_points_from_fused_groups(fused_groups_data)
            sub_workloads = self.workload.split_fusion_groups(cut_points=cut_points)

        assert len(fused_groups_data) == len(sub_workloads), (
            f"Fixed mapping has {len(fused_groups_data)} fused groups "
            f"but workload splits into {len(sub_workloads)} groups"
        )

        full_runtime_args = mapping_data.get("runtime_args", {})
        sub_mappings = []
        for i, sub_workload in enumerate(sub_workloads):
            per_group_data: dict[str, Any] = {
                "layers": mapping_data["layers"],
                "fused_groups": [fused_groups_data[i]],
                "runtime_args": self._runtime_args_for_group(full_runtime_args, sub_workload),
            }
            factory = MappingFactory(per_group_data, sub_workload, self.accelerator)
            sub_mappings.append(factory.create())

        logger.info(f"Built {len(sub_mappings)} in-memory mappings from fixed YAML: {self.mapping_path}")
        self.ctx.set(sub_workloads=sub_workloads, sub_mappings=sub_mappings)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    @staticmethod
    def _runtime_args_for_group(full_runtime_args: dict[str, Any], sub_workload) -> dict[str, Any]:
        """Select the runtime args (boundary tensors) belonging to one fusion group.

        A sub-workload's runtime args are its InEdge/OutEdge boundary nodes: model
        inputs/initializers it consumes, the output(s) it produces, and any
        inter-group fusion-boundary edge introduced by the split. Explicit layouts
        from the full mapping are preserved; boundary edges default to the standard
        (identity) layout via an empty entry.
        """
        return {
            node.name: full_runtime_args.get(node.name, {})
            for node in sub_workload.nodes
            if isinstance(node, (InEdge, OutEdge))
        }

    @staticmethod
    def _cut_points_from_fused_groups(fused_groups_data: list[dict[str, Any]]) -> list[str]:
        """Derive workload cut points from a mapping's fused-group layer assignment.

        Cut after the last layer of every group except the last one: that layer
        produces the tensor consumed by the next group, i.e. the inter-group
        fusion boundary. ``split_fusion_groups`` keeps the cut-point node in the
        preceding group, so the boundaries line up with the group definitions.
        """
        return [group["layers"][-1] for group in fused_groups_data[:-1]]

    def _parse_and_validate_yaml(self) -> dict[str, Any]:
        raw_data = open_yaml(self.mapping_path)
        validator = MappingValidator(raw_data)
        if not validator.validate():
            raise ValueError(f"Fixed mapping validation failed: {validator.errors}")
        return validator.normalized_data
