import logging
from typing import Any

from zigzag.utils import open_yaml

from stream.mapping.mapping import runtime_args_per_group
from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_validator import MappingValidator
from stream.stages.codegen.backends import AIE2_CODEGEN
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.post_ops import fold_post_ops
from stream.workload.workload import determine_fusion_cut_points

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
        self._fold_post_ops(mapping_data)
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

        runtime_args = runtime_args_per_group(mapping_data.get("runtime_args", {}), sub_workloads)
        sub_mappings = []
        for i, sub_workload in enumerate(sub_workloads):
            per_group_data: dict[str, Any] = {
                "layers": mapping_data["layers"],
                "fused_groups": [fused_groups_data[i]],
                "runtime_args": runtime_args[i],
            }
            factory = MappingFactory(per_group_data, sub_workload, self.accelerator)
            sub_mappings.append(factory.create())

        logger.info(f"Built {len(sub_mappings)} in-memory mappings from fixed YAML: {self.mapping_path}")
        self.ctx.set(sub_workloads=sub_workloads, sub_mappings=sub_mappings)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _fold_post_ops(self, mapping_data: dict[str, Any]) -> None:
        """Fold each group's post-ops into the node producing their input, which then calls their kernels after its
        own, in the order they fold. Every group is checked before the workload or a kernel changes.

        Raises:
            ValueError: A group names post-ops on an accelerator without AIE cores, or one that is not its own
                elementwise layer folding into a GEMM of its own.
        """
        groups = [group for group in mapping_data["fused_groups"] if group.get("post_ops")]
        if not groups:
            return
        if not AIE2_CODEGEN.claims(self.accelerator):
            raise ValueError(
                f"fused groups {[g['name'] for g in groups]} name post-ops, which only AIE cores price and run"
            )
        from stream.compiler.kernels.registry import POST_OP_KERNELS  # noqa: PLC0415

        layers = {layer["name"]: layer for layer in mapping_data["layers"]}
        nodes = {node.name for node in self.workload.node_list}
        workload, into = self.workload, {}
        for group in groups:
            for layer in group["post_ops"]:
                kernel = (layers.get(layer, {}).get("kernel") or {}).get("name")
                if layer not in group["layers"] or layer not in nodes or kernel not in POST_OP_KERNELS:
                    raise ValueError(
                        f"fused group {group['name']}: post-op {layer} must be one of its layers, a node of the "
                        f"workload, with an elementwise kernel ({', '.join(POST_OP_KERNELS)}), not {kernel}"
                    )
            try:
                workload, folded = fold_post_ops(workload, group["post_ops"])
            except ValueError as e:
                raise ValueError(f"fused group {group['name']}: {e}") from e
            for layer, node in folded.items():
                if node not in group["layers"] or (layers[node].get("kernel") or {}).get("name") != "gemm":
                    raise ValueError(
                        f"fused group {group['name']}: post-op {layer} folds into {node}, not a GEMM of it"
                    )
            into |= folded
        self.workload = workload
        for layer, node in into.items():
            kwargs = layers[node]["kernel"]["kwargs"]
            kwargs["post_ops"] = (*kwargs.get("post_ops", ()), layers[layer]["kernel"]["name"])
        for group in groups:
            group["layers"] = [layer for layer in group["layers"] if layer not in into]

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
