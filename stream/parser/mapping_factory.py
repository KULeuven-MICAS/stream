import inspect
from typing import Any

from stream.compiler.kernels import AIEKernels
from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping, NodeMapping
from stream.workload.workload import ComputationNode, Workload


class MappingFactory:
    def __init__(self, mapping_data: list[dict[str, Any]], workload: Workload, accelerator: Accelerator):
        self.all_mapping_data = mapping_data
        self.workload = workload
        self.accelerator = accelerator

    def create(self) -> Mapping:
        mapping = Mapping()
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
        for mapping_data in self.all_mapping_data:
            if mapping_data["name"] == node.name:
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

    def create_kernel(self, mapping_data: dict[str, Any]) -> AIEKernel:
        kernel_name = mapping_data["kernel"]["name"]
        kernel = AIEKernels.get(kernel_name, None)
        if kernel is None:
            raise ValueError(f"Unknown kernel name {kernel_name}. Available kernels: {list(AIEKernels.keys())}")
        kernel_kwargs = mapping_data["kernel"].get("kwargs", {})
        if not self.kernel_args_match_kernel_signature(kernel, kernel_kwargs):
            raise ValueError(f"Kernel arguments {kernel_kwargs} do not match kernel {kernel_name} signature.")
        return kernel(**kernel_kwargs)

    def create_inter_core_tiling(self, mapping_data: dict[str, Any], node: ComputationNode) -> Any:
        return tuple(self._convert_layer_dim_int_pair(pair, node) for pair in mapping_data["inter_core_tiling"])

    def _convert_layer_dim_int_pair(self, pair: str, node: ComputationNode):
        """Convert strings such as `D, 4` into a LayerDim and int"""
        layer_dim = LayerDim(pair.split(",")[0])
        unrolling_str = pair.split(",")[-1]
        match unrolling_str.strip(" "):
            case "all":
                unrolling = node.get_dimension_size(layer_dim)
            case "*":
                raise NotImplementedError("Wildcard unrolling '*' is not supported yet for new workload.")
                unrolling = "*"
            case _:
                unrolling = int(unrolling_str)
        return layer_dim, unrolling
