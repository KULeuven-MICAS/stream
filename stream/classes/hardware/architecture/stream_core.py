# This file is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
 
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
#===----------------------------------------------------------------------===//


from zigzag.hardware.architecture.Core import Core

from zigzag.hardware.architecture.operational_array import OperationalArray
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy

from zigzag.mapping.spatial_mapping import SpatialMapping


# Aya: Wrapping the Core class of Zigzag to extend it with more fields
class Core(Core):
    def __init__(
        self,
        core_id: int,
        operational_array: OperationalArray,
        memory_hierarchy: MemoryHierarchy,
        dataflows: SpatialMapping | None = None,
    ):
        self.core_type = 0  # Initialize it to 0 and later I change it in the accelerator_factory.. the convention is 0: Compute, 1: MemTile, 2: Offchip
        super().__init__(core_id=core_id, operational_array=operational_array, memory_hierarchy=memory_hierarchy, dataflows=dataflows)