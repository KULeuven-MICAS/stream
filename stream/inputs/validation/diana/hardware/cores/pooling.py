from stream.classes.hardware.architecture.pooling_array import PoolingArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.core import Core
from stream.classes.hardware.architecture.pooling_unit import PoolingUnit

from stream.inputs.validation.diana.hardware.cores.shared_memories import shared_l1


def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    # Fake memory level for weights (weights will be 0 bit for pooling)
    rf2 = MemoryInstance(
        name="rf_16B",
        size=16 * 8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0,
        area=0.95,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=shared_l1,
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {
                "fh": "rw_port_1",
                "tl": "rw_port_1",
                "fl": "rw_port_2",
                "th": "rw_port_2",
            },
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=rf2,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    return memory_hierarchy_graph


def get_operational_array():
    pooling_unit_input_precision = [8, 8]
    pooling_energy = 0.1
    pooling_area = 0.01
    dimensions = {"D1": 16, "D2": 2, "D3": 2}
    pooling_unit = PoolingUnit(
        pooling_unit_input_precision, pooling_energy, pooling_area
    )
    pooling_array = PoolingArray(pooling_unit, dimensions)
    return pooling_array


def get_dataflows():
    return [{"D1": ("OX", 16), "D2": ("FX", 3), "D3": ("FY", 3)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
