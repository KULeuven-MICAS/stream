import os
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from stream.inputs.exploration.hardware.memory_pool.Stream_journal_exploration import *


def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_64B_1r1w_8b(),
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 0, 0)},
    )

    memory_hierarchy_graph.add_memory(
        memory_instance=reg_256B_2r2w_16b(),
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
        ),
        served_dimensions={(1, 0, 0), (0, 1, 0)},
    )

    memory_hierarchy_graph.add_memory(
        memory_instance=reg_256B_1r1w_8b(),
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 0, 1)},
    )

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_32KB_2_16KB_1r1w_224b(),
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
        ),
        served_dimensions="all",
    )

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_2_128KB_1r1w_36b(),
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_2_128KB_1r1w_128b(),
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
        ),
        served_dimensions="all",
    )

    ####################################################################################################################

    from zigzag.visualization.graph.memory_hierarchy import (
        visualize_memory_hierarchy_graph,
    )

    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def get_operational_array():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.04
    multiplier_area = 1
    dimensions = {"D1": 3, "D2": 3, "D3": 28}
    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_dataflows():
    return [{"D1": ("FX", 3), "D2": ("FY", 3), "D3": ("OX", 28)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
