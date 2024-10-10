from dataclasses import dataclass

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

INTRA_CORE_MAPPING_DEFAULT = IntraCoreMappingAttributes(
    spatial_mapping=SpatialMapping.empty(),
    spatial_mapping_hint=SpatialMappingHint.empty(),
    core_allocation=[0],
    core_allocation_is_fixed=False,
    temporal_ordering=LayerTemporalOrdering.empty(),
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
    core_allocation_is_fixed: bool
    intra_core_tiling: list[tuple[LayerDim, int]]
    inter_core_tiling: list[tuple[LayerDim, int]]
