from typing import Any

from zigzag.datatypes import (
    LayerDim,
    OADimension,
    UnrollFactor,
)
from zigzag.mapping.spatial_mapping import (
    MappingSingleOADim,
    SpatialMapping,
)

from stream.workload.mapping import TILING_T, TILING_WILDCARD_T, InterCoreMappingAttributes


class MappingFactory:
    def __init__(self, mapping_data: list[dict[str, Any]]):
        self.all_mapping_data = mapping_data

    def create(self) -> dict[str, InterCoreMappingAttributes]:  # type: ignore
        all_mappings: dict[str, InterCoreMappingAttributes] = {}

        for mapping_data in self.all_mapping_data:
            op_type = mapping_data["name"]
            core_allocation = mapping_data["core_allocation"]
            layer_dimension_names = mapping_data["layer_dimension_names"]
            spatial_mapping = self.create_spatial_mapping(mapping_data)
            inter_core_tiling = self.create_inter_core_tiling(mapping_data)
            intra_core_tiling = self.create_intra_core_tiling(mapping_data)
            mapping = InterCoreMappingAttributes(
                op_type=op_type,
                spatial_mapping=spatial_mapping,
                core_allocation=core_allocation,
                inter_core_tiling=inter_core_tiling,
                intra_core_tiling=intra_core_tiling,
                layer_dimension_names=layer_dimension_names,
            )
            all_mappings[op_type] = mapping
        return all_mappings

    def create_spatial_mapping(self, mapping_data: dict[str, Any]) -> SpatialMapping:
        if mapping_data["spatial_mapping"] is None:
            return SpatialMapping.empty()

        user_data: dict[str, list[str]] = mapping_data["spatial_mapping"]
        spatial_mapping_dict: dict[OADimension, MappingSingleOADim] = {}

        for oa_dim_str, unrolling_list in user_data.items():
            oa_dim = OADimension(oa_dim_str)
            mapping_this_oa_dim = self.create_mapping_single_oa_dim(unrolling_list)
            spatial_mapping_dict[oa_dim] = mapping_this_oa_dim

        return SpatialMapping(spatial_mapping_dict)

    def create_mapping_single_oa_dim(self, mapping_data: list[str]) -> MappingSingleOADim:
        mapping_dict: dict[LayerDim, UnrollFactor] = {}

        for single_unrolling in mapping_data:
            layer_dim, unrolling = self.__convert_layer_dim_int_pair(single_unrolling)
            mapping_dict[layer_dim] = unrolling  # type: ignore

        return MappingSingleOADim(mapping_dict)

    def create_inter_core_tiling(self, mapping_data: dict[str, Any]) -> TILING_WILDCARD_T:
        return [self.__convert_layer_dim_int_pair(pair) for pair in mapping_data["inter_core_tiling"]]

    def create_intra_core_tiling(self, mapping_data: dict[str, Any]) -> TILING_T:
        return [self.__convert_layer_dim_int_pair(pair) for pair in mapping_data["intra_core_tiling"]]  # type: ignore

    def __convert_layer_dim_int_pair(self, pair: str):
        """Convert strings such as `D, 4` into a LayerDim and int"""
        layer_dim_str = pair.split(",")[0]
        unrolling_str = pair.split(",")[-1]
        match unrolling_str.strip(" "):
            case "all":
                unrolling = "all"
            case "*":
                unrolling = "*"
            case _:
                unrolling = int(unrolling_str)
        layer_dim = LayerDim(layer_dim_str)
        return layer_dim, unrolling
