import os
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.operational_unit import Multiplier
from zigzag.hardware.architecture.operational_array import MultiplierArray
from zigzag.hardware.architecture.memory_instance import MemoryInstance
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core


def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """

    reg_W_64B = MemoryInstance(
        name="rf_64B",
        size=64 * 8,
        r_bw=8,
        w_bw=8,
        r_cost=0.053,
        w_cost=0.053,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O_1K = MemoryInstance(
        name="rf_1KB",
        size=1024 * 8,
        r_bw=16,
        w_bw=16,
        r_cost=0.54,
        w_cost=0.6,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    sram_64K_with_16_4K_bank_128_1r_1w = MemoryInstance(
        name="sram_64KB_A",
        size=4096 * 16 * 8,
        r_bw=128 * 16,
        w_bw=128 * 16,
        r_cost=2.31 * 16,
        w_cost=5.93 * 16,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )

    sram_1M_with_8_128K_bank_128_1r_1w_A = MemoryInstance(
        name="sram_1MB_A",
        size=131072 * 8 * 8,
        r_bw=128 * 8,
        w_bw=128 * 8,
        r_cost=26.01 * 8,
        w_cost=23.65 * 8,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )

    sram_1M_with_8_128K_bank_128_1r_1w_W = MemoryInstance(
        name="sram_1MB_W",
        size=131072 * 8 * 8,
        r_bw=128 * 8,
        w_bw=128 * 8,
        r_cost=26.01 * 8,
        w_cost=23.65 * 8,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )

    #######################################################################################################################

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_W_64B,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 0)},
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O_1K,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
        ),
        served_dimensions={(0, 1)},
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_64K_with_16_4K_bank_128_1r_1w,
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
        ),
        served_dimensions="all",
    )

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_W,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_A,
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
    dimensions = {"D1": 32, "D2": 32}  # {'D1': ('K', 32), 'D2': ('C', 32)}
    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_dataflows():
    return [{"D1": ("K", 32), "D2": ("C", 32)}, {"D1": ("G", 32)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
