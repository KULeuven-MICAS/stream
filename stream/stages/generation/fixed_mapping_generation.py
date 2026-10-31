import logging
from typing import Any

from zigzag.utils import open_yaml

from stream.mapping.mapping import edge_tensor
from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.tensor import Tensor
from stream.workload.workload import InEdge, OutEdge, determine_fusion_cut_points

logger = logging.getLogger(__name__)


class FixedMappingGenerationStage(Stage):
    """Parse a fixed mapping YAML and build per-group Mappings in-memory.

    Reads the mapping YAML once, splits the workload into fusion groups,
    then builds a scoped Mapping per group via MappingFactory — no intermediate
    YAML files are written or read.
    """

    reads = ("accelerator", "workload", "mapping_path")
    writes = ("sub_workloads", "sub_mappings")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.get("accelerator")
        self.workload = self.ctx.get("workload")
        self.mapping_path: str = self.ctx.get("mapping_path")

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
        declared = self._declared_per_tensor(full_runtime_args, sub_workloads)
        sub_mappings = []
        for i, sub_workload in enumerate(sub_workloads):
            per_group_data: dict[str, Any] = {
                "layers": mapping_data["layers"],
                "fused_groups": [fused_groups_data[i]],
                "runtime_args": self._runtime_args_for_group(full_runtime_args, sub_workload, declared),
            }
            factory = MappingFactory(per_group_data, sub_workload, self.accelerator)
            sub_mappings.append(factory.create())

        logger.info(f"Built {len(sub_mappings)} in-memory mappings from fixed YAML: {self.mapping_path}")
        self.ctx.set(sub_workloads=sub_workloads, sub_mappings=sub_mappings)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    @staticmethod
    def _runtime_args_for_group(
        full_runtime_args: dict[str, Any], sub_workload, declared: dict[Tensor, dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Select the runtime args (boundary tensors) belonging to one fusion group.

        A sub-workload's runtime args are its InEdge/OutEdge boundary nodes: model
        inputs/initializers it consumes, the output(s) it produces, and any
        inter-group fusion-boundary edge introduced by the split. Explicit layouts
        from the full mapping are preserved; the two ends of a fusion cut both take
        the entry ``declared`` for the cut's tensor, merged over its ends, else the
        standard (identity) layout via an empty entry.
        """
        declared = declared or {}
        return {
            node.name: declared.get(edge_tensor(node), full_runtime_args.get(node.name, {}))
            for node in sub_workload.nodes
            if isinstance(node, (InEdge, OutEdge))
        }

    @staticmethod
    def _declared_per_tensor(full_runtime_args: dict[str, Any], sub_workloads) -> dict[Tensor, dict[str, Any]]:
        """The runtime args declared for each edge tensor, merged over its edges; the two ends of a fusion cut must not
        declare one key differently."""
        declared: dict[Tensor, dict[str, tuple[str, Any]]] = {}
        for node in (node for sub_workload in sub_workloads for node in sub_workload.nodes):
            if not isinstance(node, (InEdge, OutEdge)) or node.name not in full_runtime_args:
                continue
            merged = declared.setdefault(edge_tensor(node), {})
            for key, value in full_runtime_args[node.name].items():
                name, first = merged.setdefault(key, (node.name, value))
                if first != value:
                    raise ValueError(
                        f"{name} and {node.name} are the two ends of one fusion cut but declare {key} {first} and "
                        f"{value}"
                    )
        return {tensor: {key: value for key, (_, value) in merged.items()} for tensor, merged in declared.items()}

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
