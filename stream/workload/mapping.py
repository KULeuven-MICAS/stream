from dataclasses import dataclass
from typing import Literal, TypeAlias

from zigzag.datatypes import (
    Constants,
    LayerDim,
)
from zigzag.mapping.spatial_mapping import SpatialMapping, SpatialMappingHint
from zigzag.workload.layer_attributes import (
    LayerTemporalOrdering,
    MemoryOperandLinks,
)
from zigzag.workload.layer_node import MappingAttributes as IntraCoreMappingAttributes

TILING_T: TypeAlias = list[tuple[LayerDim, int]]
TILING_WILDCARD_T: TypeAlias = list[tuple[LayerDim, int | Literal["*", "all"]]]

INTRA_CORE_MAPPING_DEFAULT = IntraCoreMappingAttributes(
    spatial_mapping=SpatialMapping.empty(),
    spatial_mapping_hint=SpatialMappingHint.empty(),
    temporal_ordering=LayerTemporalOrdering.empty(),  # type: ignore
    memory_operand_links=MemoryOperandLinks(
        {
            Constants.LAYER_OP_I: Constants.MEM_OP_1,
            Constants.LAYER_OP_W: Constants.MEM_OP_2,
            Constants.OUTPUT_LAYER_OP: Constants.OUTPUT_MEM_OP,
        }
    ),
)


@dataclass
class InterCoreMappingAttributes:
    op_type: str
    spatial_mapping: SpatialMapping
    core_allocation: list[int]
    intra_core_tiling: TILING_T
    inter_core_tiling: TILING_WILDCARD_T | TILING_T
    layer_dimension_names: list[str]
