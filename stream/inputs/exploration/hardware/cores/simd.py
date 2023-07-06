from stream.classes.hardware.architecture.simd_array import SimdArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.core import Core
from stream.classes.hardware.architecture.simd_unit import SimdUnit
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
        memory_instance=sram_256KB_2_128KB_1r1w_256b(),
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
        ),
        served_dimensions="all",
    )

    # from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def get_operational_array():
    add_unit_input_precision = [8, 8]
    add_energy = 0.01
    add_area = 0.01
    dimensions = {"D1": 32}
    add_unit = SimdUnit(add_unit_input_precision, add_energy, add_area)
    add_array = SimdArray(add_unit, dimensions)
    return add_array


def get_dataflows():
    return [{"D1": ("K", 32)}, {"D1": ("OX", 32)}, {"D1": ("G", 32)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
