from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.core import Core

from stream.inputs.validation.diana.hardware.cores.shared_memories import shared_l1


def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=#bit"""
    # Defintion of register file for inputs
    rf_1B_I = MemoryInstance(
        name="rf_1B",
        mem_type="rf",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=1,
        w_cost=1.2,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
    )
    # Defintion of register file for weights
    rf_1B_W = MemoryInstance(
        name="rf_1B",
        mem_type="rf",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=1,
        w_cost=1.2,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
    )
    # Defintion of rRegister file for outputs
    rf_2B = MemoryInstance(
        name="rf_4B",
        size=16,
        r_bw=16,
        w_bw=16,
        r_cost=3,
        w_cost=3.6,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
    )
    # Defintion of first SRAM for weights
    l1_w = MemoryInstance(
        name="l1_w",
        mem_type="sram",
        size=64 * 1024 * 8,
        r_bw=16 * 8,
        w_bw=16 * 8,
        r_cost=50,
        w_cost=55,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high = 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high = 
    tl: to low = rd_out_to_low = 
    """
    # Register file for input
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_1B_I,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 1)},
    )
    # Register file for weight
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_1B_W,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(1, 0)},
    )
    # Register file for output
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_2B,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
        ),
        served_dimensions=set(),
    )

    # First SRAM for weights
    memory_hierarchy_graph.add_memory(
        memory_instance=l1_w,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    # First SRAM for inputs and outputs
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

    return memory_hierarchy_graph


def get_operational_array():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.04
    multiplier_area = 1
    dimensions = {"D1": 16, "D2": 16}  # {'D1': ('OX', 16), 'D2': ('K', 16)}

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_dataflows():
    return [{"D1": ("OX", 16), "D2": ("K", 16)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core


if __name__ == "__main__":
    print(get_core(1))
