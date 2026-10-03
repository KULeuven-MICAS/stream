from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

from xdsl.context import Context
from xdsl.dialects.builtin import AffineMapAttr
from xdsl.ir.affine import AffineMap
from xdsl.parser import Parser

from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping
from stream.workload.node import OutEdge
from stream.workload.workload import ComputationNode, InEdge, Workload

if TYPE_CHECKING:
    # AIE kernels live in stream.compiler (optional AIE install). Imported lazily inside
    # create_kernel so the base mapping path stays free of the AIE toolchain.
    from stream.compiler.kernels.aie_kernel import AIEKernel


class MappingFactory:
    def __init__(
        self,
        mapping_data: dict[str, Any],
        workload: Workload,
        accelerator: Accelerator,
    ):
        self.layers_data: list[dict[str, Any]] = mapping_data.get("layers", [])
        self.fused_groups_data: list[dict[str, Any]] = mapping_data.get("fused_groups", [])
        self.runtime_args_data: dict[str, str] = mapping_data.get("runtime_args", {})
        self.workload = workload
        self.accelerator = accelerator

    def create(self) -> Mapping:
        mapping = Mapping(fused_groups=self.create_fused_groups(), runtime_args=self.create_runtime_args())
        # For each computation node in the graph set up mapping attributes
        for cn in self.workload.get_computation_nodes():
            mapping_data = self.get_mapping_data_for_node(cn)
            resource_allocation = self.get_resource_allocation(mapping_data)
            inter_core_tiling = self.convert_inter_core_tiling(mapping_data, node=cn)
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

    def get_resource_allocation(self, mapping_data: dict[str, Any]) -> tuple[tuple[Core, ...], ...]:
        cores = []
        for core_ids in mapping_data["core_allocation"]:
            core_group = []
            for core_id in core_ids:
                core = self.accelerator.get_core(core_id)
                if core is None:
                    raise ValueError(f"Core with id {core_id} not found in accelerator.")
                core_group.append(core)
            cores.append(tuple(core_group))
        return tuple(cores)

    def create_kernel(self, mapping_data: dict[str, Any]) -> AIEKernel | None:
        kernel_name = (mapping_data.get("kernel") or {}).get("name")
        if not kernel_name:
            return None
        try:
            from stream.compiler.kernels.registry import AIE_KERNELS  # noqa: PLC0415
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"The mapping names kernel {kernel_name!r}, but the AIE kernels cannot be imported ({e}); "
                "install them with `stream-setup-aie`."
            ) from e

        if kernel_name not in AIE_KERNELS:
            raise ValueError(f"Unknown kernel name {kernel_name!r}. Known kernels: {sorted(AIE_KERNELS)}")
        kernel_kwargs = mapping_data["kernel"].get("kwargs", {})
        try:
            kernel = AIE_KERNELS[kernel_name](**kernel_kwargs, library=self.accelerator.kernel_library)
        except TypeError as e:
            raise ValueError(f"Kernel arguments {kernel_kwargs} do not match kernel {kernel_name}: {e}") from e
        if kernel.library is not None:
            kernel.validate()
        return kernel

    def convert_inter_core_tiling(
        self, mapping_data: dict[str, Any], node: ComputationNode
    ) -> tuple[InterCoreTiling, ...]:
        entries = mapping_data.get("inter_core_tiling", [])
        converted_entries = []
        for entry in entries:
            converted = tuple(self._convert_inter_core_tiling_entry(sub_entry, node) for sub_entry in entry)
            converted_entries.append(converted)
        return tuple(converted_entries)

    def _convert_inter_core_tiling_entry(self, entry: dict[str, Any], node: ComputationNode) -> tuple[LayerDim, int]:
        dim_str = entry["dim"]
        if not isinstance(dim_str, str) or not dim_str.startswith("D"):
            raise ValueError(f"Unsupported inter_core_tiling dimension format: {dim_str!r} for node {node.name}")
        layer_dim = LayerDim(position=int(dim_str[1:]), prefix="d")
        split_val = int(entry["split"])
        return layer_dim, split_val

    def create_fused_groups(self) -> list[FusedGroup]:
        fused_groups: list[FusedGroup] = []
        for fused_group in self.fused_groups_data:
            layers = tuple(fused_group.get("layers", []))
            intra_core_tiling = fused_group.get("intra_core_tiling", []) or []
            if intra_core_tiling:
                tiling = tuple(self._convert_intra_core_tiling_entry(entry) for entry in intra_core_tiling)
            else:
                tiling = self._call_tile_tiling(layers)
            fused_groups.append(
                FusedGroup(
                    name=fused_group["name"],
                    layers=layers,
                    intra_core_tiling=tiling,
                    runtime_dims=self._runtime_dims(layers, tiling),
                ),
            )
        return fused_groups

    def _group_kernels(self, layers: tuple[str, ...]):
        for name in layers:
            node = self.workload.get_node_by_name(name)
            if isinstance(node, ComputationNode):
                kernel = self.create_kernel(self.get_mapping_data_for_node(node))
                if kernel is not None:
                    yield node, kernel

    def _call_tile_tiling(self, layers: tuple[str, ...]) -> tuple[tuple[LayerDim, int], ...]:
        """A group's default tiling, innermost first: its kernels' call tiles, with the dimensions its layers
        share innermost, then one at a time each dimension no kernel addresses, such as a batch axis."""
        tiling: dict[LayerDim, int] = {}
        addressed: set[LayerDim] = set()
        kernels = list(self._group_kernels(layers))
        for node, kernel in kernels:
            node_dims = self.workload.get_dims(node)
            for position, size, dim in kernel.call_tile(node):
                layer_dim = node_dims[position]
                addressed.add(layer_dim)
                extent = self.workload.get_dimension_size(layer_dim)
                if size < extent or dim.keep_whole:
                    tiling.setdefault(layer_dim, min(size, extent))
        shared = dict.fromkeys(tiling, 0)
        for a, b in pairwise(set(self.workload.get_dims(node)) for node, _ in kernels):
            for dim in a & b:
                if dim in shared:
                    shared[dim] += 1
        batch = dict.fromkeys(
            dim
            for node, _ in kernels
            for dim in self.workload.get_dims(node)
            if dim not in addressed and self.workload.get_dimension_size(dim) > 1
        )
        return (*sorted(tiling.items(), key=lambda item: shared[item[0]]), *((dim, 1) for dim in batch))

    def _runtime_dims(self, layers: tuple[str, ...], tiling) -> tuple[LayerDim, ...]:
        """Tiling dimensions every kernel of the group takes as a runtime size."""
        compiled: set[LayerDim] = set()
        runtime: set[LayerDim] = set()
        for node, kernel in self._group_kernels(layers):
            node_dims = self.workload.get_dims(node)
            for position, _, dim in kernel.call_tile(node):
                (runtime if dim.runtime else compiled).add(node_dims[position])
        return tuple(dim for dim, _ in tiling if dim in runtime and dim not in compiled)

    def _convert_intra_core_tiling_entry(self, entry: dict[str, Any]) -> tuple[LayerDim, int]:
        node_name, dim_name = entry["dim"].rsplit(".", 1)
        assert dim_name[0] == "D", f"Unsupported intra_core_tiling dimension format: {entry['dim']!r}"
        dim_idx = int(dim_name[1:])
        node = self.workload.get_node_by_name(node_name)
        assert isinstance(node, ComputationNode), f"Node {node_name} not found in workload."
        dim = self.workload.get_dims(node)[dim_idx]
        assert isinstance(dim, LayerDim), f"Dimension at index {dim_idx} of node {node_name} is not a LayerDim."
        return dim, int(entry["tile"])

    def create_runtime_args(self) -> dict[str, str]:
        runtime_args = {}
        for tensor_name, args in self.runtime_args_data.items():
            if not isinstance(args, dict):
                raise ValueError(f"Runtime args for tensor {tensor_name} must be a dict.")
            layout = args.get("layout", None)
            affine_map = self._get_affine_map(layout) if layout else self._get_standard_layout(tensor_name)
            runtime_args[tensor_name] = affine_map
        return runtime_args

    def _get_standard_layout(self, tensor_name: str) -> AffineMap:
        node = self.workload.get_node_by_name(tensor_name)
        if isinstance(node, InEdge):
            tensor = node.outputs[0]
        elif isinstance(node, OutEdge):
            tensor = node.inputs[0]
        else:
            raise NotImplementedError(f"Standard layout inference not implemented for tensor {tensor_name}.")
        num_dims = len(tensor.shape)
        map = AffineMap.identity(num_dims)
        return map

    def _get_affine_map(self, layout: str) -> AffineMap:
        """Layout in the form of (d0, d1, ...) -> (d1, d0, ...) gets converted to AffineMap."""
        # Prepend affine_map< and append >
        layout_str = f"affine_map<{layout}>"
        ctx = Context()
        parser = Parser(ctx, layout_str)
        attribute = parser.parse_attribute()
        assert isinstance(attribute, AffineMapAttr)
        return attribute.data
