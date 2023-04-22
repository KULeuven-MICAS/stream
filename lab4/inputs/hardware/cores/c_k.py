from zigzag.classes.hardware.architecture.core import Core

import lab4.inputs.hardware.cores.core_definition as core_def
import lab4.inputs.hardware.cores.core_description as core_desc


def get_dataflows(quad_core=False):
    if quad_core:
        return [
            {
                "D1": ("C", core_def.quad_core_multiplier_array_size_2D[0]),
                "D2": ("K", core_def.quad_core_multiplier_array_size_2D[1]),
            },
            {"D1": ("G", core_def.quad_core_multiplier_array_size_2D[0])},
        ]
    else:
        return [
            {
                "D1": ("C", core_def.single_core_multiplier_array_size_2D[0]),
                "D2": ("K", core_def.single_core_multiplier_array_size_2D[1]),
            },
            {"D1": ("G", core_def.single_core_multiplier_array_size_2D[0])},
        ]


def get_core(id, quad_core):
    operational_array = core_desc.get_multiplier_array_2D(quad_core)
    memory_hierarchy = core_desc.get_memory_hierarchy_C_K_dataflow(operational_array)
    dataflows = get_dataflows(quad_core)
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
