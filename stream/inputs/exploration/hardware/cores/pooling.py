from stream.classes.hardware.architecture.pooling_array import PoolingArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.core import Core
from stream.classes.hardware.architecture.pooling_unit import PoolingUnit
from stream.inputs.exploration.hardware.memory_pool.Stream_journal_exploration import *

def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_2_128KB_1r1w_36b(),
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_64B_1r1w_8b(),
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )

    # from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def get_operational_array():
    pooling_unit_input_precision = [8, 8]
    pooling_energy = 0.1
    pooling_area = 0.01
    dimensions = {"D1": 3, "D2": 3}
    pooling_unit = PoolingUnit(
        pooling_unit_input_precision, pooling_energy, pooling_area
    )
    pooling_array = PoolingArray(pooling_unit, dimensions)
    return pooling_array


def get_dataflows():
    return [{"D1": ("FX", 3), "D2": ("FY", 3)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
