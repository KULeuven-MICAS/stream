# This file is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
 
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
#===----------------------------------------------------------------------===//


from zigzag.classes.hardware.architecture.core import Core, MemoryHierarchy, OperationalArray


# Aya: Wrapping the Core class of Zigzag to extend it with more fields
class Core(Core):
    def __init__(
        self,
        id: int,
        operational_array: OperationalArray,
        memory_hierarchy: MemoryHierarchy,
        core_type: int = 0, # Aya: the convention is 0: Compute, 1: MemTile, 2: Offchip
        dataflows: list = None,
    ):
        self.core_type = core_type
        super().__init__(id, operational_array, memory_hierarchy, dataflows)