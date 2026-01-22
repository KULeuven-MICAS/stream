import inspect
from typing import Any

from stream.compiler.kernels import AIEKernels
from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping
from stream.workload.workload import ComputationNode, Workload


class MappingFactory:
    def __init__(self, mapping_data: dict[str, Any], workload: Workload, accelerator: Accelerator):
        self.layers_data: list[dict[str, Any]] = mapping_data.get("layers", [])
        self.fused_groups_data: list[dict[str, Any]] = mapping_data.get("fused_groups", [])
        self.workload = workload
        self.accelerator = accelerator

    def create(self) -> Mapping:
        mapping = Mapping(fused_groups=self.create_fused_groups())
        # For each computation node in the graph set up mapping attributes
        for cn in self.workload.get_computation_nodes():
            mapping_data = self.get_mapping_data_for_node(cn)
            resource_allocation = tuple(self.get_resource_allocation(mapping_data))
            inter_core_tiling = self.create_inter_core_tiling(mapping_data, node=cn)
            kernel = self.create_kernel(mapping_data)
            mapping.set(
                cn,
                NodeMapping(
                    resource_allocation=resource_allocation,
                    inter_core_tiling=inter_core_tiling,
                    kernel=kernel,
                ),
            )
        return mapping

    def get_mapping_data_for_node(self, node: Any) -> dict[str, Any]:
        for mapping_data in self.layers_data:
            if mapping_data["name"] in [node.name, node.type]:
                return mapping_data
        raise ValueError(f"No mapping data found for node with name {node.name}")

    def get_resource_allocation(self, mapping_data: dict[str, Any]) -> list["Core"]:
        core_ids = mapping_data["core_allocation"]
        cores = []
        for core_id in core_ids:
            core = self.accelerator.get_core(core_id)
            if core is None:
                raise ValueError(f"Core with id {core_id} not found in accelerator.")
            cores.append(core)
        return cores

    def kernel_args_match_kernel_signature(self, kernel, kwargs):
        try:
            inspect.signature(kernel).bind(**kwargs)
            return True
        except TypeError:
            return False

    def create_kernel(self, mapping_data: dict[str, Any]) -> AIEKernel | None:
        kernel_name = mapping_data["kernel"]["name"]
        kernel = AIEKernels.get(kernel_name, None)
        if kernel is None:
            return None
        kernel_kwargs = mapping_data["kernel"].get("kwargs", {})
        if not self.kernel_args_match_kernel_signature(kernel, kernel_kwargs):
            raise ValueError(f"Kernel arguments {kernel_kwargs} do not match kernel {kernel_name} signature.")
        return kernel(**kernel_kwargs)

    def create_inter_core_tiling(self, mapping_data: dict[str, Any], node: ComputationNode) -> InterCoreTiling:
        entries = mapping_data.get("inter_core_tiling", [])
        return tuple(self._convert_inter_core_tiling_entry(entry, node) for entry in entries)

    def _convert_inter_core_tiling_entry(self, entry: dict[str, Any], node: ComputationNode) -> tuple[LayerDim, int]:
        dim_str = entry["dim"]
        if not isinstance(dim_str, str) or not dim_str.startswith("D"):
            raise ValueError(f"Unsupported inter_core_tiling dimension format: {dim_str!r} for node {node.name}")
        layer_dim = LayerDim(int(dim_str[1:]))
        split_val = int(entry["split"])
        return layer_dim, split_val

    def create_fused_groups(self) -> list[FusedGroup]:
        fused_groups: list[FusedGroup] = []
        for fused_group in self.fused_groups_data:
            intra_core_tiling = fused_group.get("intra_core_tiling", []) or []
            fused_groups.append(
                FusedGroup(
                    name=fused_group["name"],
                    layers=tuple(fused_group.get("layers", [])),
                    intra_core_tiling=tuple(
                        self._convert_intra_core_tiling_entry(entry) for entry in intra_core_tiling
                    ),
                ),
            )
        return fused_groups

    def _convert_intra_core_tiling_entry(self, entry: dict[str, Any]) -> tuple[LayerDim, int]:
        node_name, dim_name = entry["dim"].split(".")
        assert dim_name[0] == "D", f"Unsupported intra_core_tiling dimension format: {entry['dim']!r}"
        dim_idx = int(dim_name[1:])
        node = self.workload.get_node_by_name(node_name)
        assert isinstance(node, ComputationNode), f"Node {node_name} not found in workload."
        dim = self.workload.get_dims(node)[dim_idx]
        assert isinstance(dim, LayerDim), f"Dimension at index {dim_idx} of node {node_name} is not a LayerDim."
        return dim, int(entry["tile"])
