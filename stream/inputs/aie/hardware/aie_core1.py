import os
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.core import Core

from stream.inputs.aie.hardware.mem_tile import mem_tile
from stream.inputs.aie.hardware.mem_tile import mem_tile1


def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
   

    rf_iw = MemoryInstance(
        name="rf_iw",
        size=512*12, #12x512b X registers
        r_bw=512*12,
        w_bw=512*12,
        r_cost=0.01,
        w_cost=0.01,
        area=0,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=0,
    )  #
    rf_i = MemoryInstance(
        name="rf_i",
        size=512*6, #12x512b X registers
        r_bw=512*6,
        w_bw=512*6,
        r_cost=0.01,
        w_cost=0.01,
        area=0,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=0,
    )  # rd E per bit 0.125
    
    rf_w = MemoryInstance(
        name="rf_w",
        size=512*6, #12x512b X registers
        r_bw=512*6,
        w_bw=512*6,
        r_cost=0.01,
        w_cost=0.01,
        area=0,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=0,
    )  # rd E per bit 0.125
    # Defintion of register file for outputs
    rf_o = MemoryInstance(
        name="rf_o",
        size=2048*5, #5x2048b X registers
        r_bw=2048*5,
        w_bw=2048*5,
        r_cost=0.01,
        w_cost=0.01,
        area=0,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=0,
    )  # rd E per bit 0.0625
    # Defintion of first SRAM for input and outputs
    l1_oiw = MemoryInstance(
        name="l1_oiw",
        size=64*1024*8,
        r_bw=512,
        w_bw=512,
        r_cost=1,
        w_cost=1,
        area=0,
        r_port=2,
        w_port=1,
        # rw_port=2,
        latency=1,
    )  # rd E per bit 0.08

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    # memory_hierarchy_graph.add_memory(
    #     memory_instance=rf_i,
    #     operands=("I1",),
    #     port_alloc=({"fh": "rw_port_1", "tl": "rw_port_2", "fl": None, "th": None},
    #     # {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
    #     # {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
    #     ),
    #     served_dimensions='all',
    # )
    # memory_hierarchy_graph.add_memory(
    #     memory_instance=rf_w,
    #     operands=("I2",),
    #     port_alloc=({"fh": "rw_port_1", "tl": "rw_port_2", "fl": None, "th": None},
    #     # {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
    #     # {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
    #     ),
    #     served_dimensions='all',
    # )
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_iw,
        operands=("I1", "I2",),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_2", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_2", "fl": None, "th": None},
        
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_o,
        operands=("O"),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_2", "fl": "rw_port_1", "th": "rw_port_2"},
        ),
        served_dimensions='all',
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=l1_oiw,
        operands=("O","I1","I2"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_2"},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
           {"fh": "w_port_1", "tl": "r_port_2", "fl": None, "th": None}
        ),
        served_dimensions='all',
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=mem_tile1,
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
        ),
        served_dimensions='all',
    )


    from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def get_operational_array():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0
    vec_macs=1
    # dimensions = {"D1": 256} 
    dimensions = {'D1': 16, 'D2': 16}

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_dataflows():
    return [
            {'D1': ('K', 16), 'D2': ('C', 16)},
            # {"D1": ("C", 256)},
            # {"D1": ("K", 256)},
            # {"D1": ("OX", 256)},
            # {"D1": ("OY", 256)},
            {"D1": ("G", 256)},
            ]
    # return [{'D1': ('K', 16), 'D2': ('C', 16), 'D3': ('OX', 4), 'D4': ('FX', 3)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core


if __name__ == "__main__":
    print(get_core(1))