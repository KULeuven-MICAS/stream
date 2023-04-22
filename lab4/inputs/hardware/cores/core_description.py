from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy

import lab4.inputs.hardware.cores.core_definition as core_def


def get_multiplier_array_2D():
    """Multiplier array variables"""
    multiplier_input_precision = [
        core_def.operand_precision,
        core_def.operand_precision,
    ]
    multiplier_energy = core_def.energy_mac_operation
    multiplier_area = 0
    dimensions = {
        "D1": core_def.quad_core_multiplier_array_size_2D[0],
        "D2": core_def.quad_core_multiplier_array_size_2D[1],
    }

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_multiplier_array_3D():
    """Multiplier array variables"""
    multiplier_input_precision = [
        core_def.operand_precision,
        core_def.operand_precision,
    ]
    multiplier_energy = core_def.energy_mac_operation
    multiplier_area = 0
    dimensions = {
        "D1": core_def.quad_core_multiplier_array_size_3D[0],
        "D2": core_def.quad_core_multiplier_array_size_2D[1],
        "D3": core_def.quad_core_multiplier_array_size_3D[2],
    }

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_memory_hierarchy_C_K_dataflow(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=#bit"""
    # Defintion of register file for inputs and weights
    rf_1B = MemoryInstance(
        name="rf_1B",
        mem_type="rf",
        size=core_def.size_rf_weight_input * 8,
        r_bw=core_def.width_rf_weight_input,
        r_port=1,
        w_port=1,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of register file for outputs
    rf_2B = MemoryInstance(
        name="rf_4B",
        mem_type="rf",
        size=core_def.size_rf_outputs * 8,
        r_bw=core_def.width_rf_outputs,
        r_port=2,
        w_port=2,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for weights
    l1_w = MemoryInstance(
        name="l1_w",
        mem_type="sram",
        size=core_def.size_l1_weights * 8,
        r_bw=core_def.width_l1_weights,
        r_port=1,
        w_port=1,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for inputs and outputs
    l1_io = MemoryInstance(
        name="l1_io",
        mem_type="sram",
        size=core_def.size_l1_activation * 8,
        r_bw=core_def.width_l1_activation,
        r_port=0,
        w_port=0,
        rw_port=2,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for weights
    l2_w = MemoryInstance(
        name="l2_w",
        mem_type="sram",
        size=core_def.size_l2_weights * 8,
        r_bw=core_def.width_l2_weigths,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for inputs and outputs
    l2_io = MemoryInstance(
        name="l2_io",
        mem_type="sram",
        size=core_def.size_l2_activation * 8,
        r_bw=core_def.width_l2_activation,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=1,
        auto_cost_extraction=True,
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
        memory_instance=rf_1B,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 1)},
    )
    # Register file for weight
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_1B,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    # Register file for output
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_2B,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
        ),
        served_dimensions={(1, 0)},
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
        memory_instance=l1_io,
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
    # Second SRAM for weights
    memory_hierarchy_graph.add_memory(
        memory_instance=l2_w,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    # Second SRAM for inputs and output
    memory_hierarchy_graph.add_memory(
        memory_instance=l2_io,
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


def get_memory_hierarchy_OX_FX_FY_dataflow(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=#bit"""
    # Defintion of register file for inputs and weights
    rf_1B = MemoryInstance(
        name="rf_1B",
        mem_type="rf",
        size=core_def.size_rf_weight_input * 8,
        r_bw=core_def.width_rf_weight_input,
        r_port=1,
        w_port=1,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of rRegister file for outputs
    rf_2B = MemoryInstance(
        name="rf_4B",
        mem_type="rf",
        size=core_def.size_rf_outputs * 8,
        r_bw=core_def.width_rf_outputs,
        r_port=2,
        w_port=2,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for weights
    l1_w = MemoryInstance(
        name="l1_w",
        mem_type="sram",
        size=core_def.size_l1_weights * 8,
        r_bw=core_def.width_l1_weights,
        r_port=1,
        w_port=1,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for inputs and outputs
    l1_io = MemoryInstance(
        name="l1_io",
        mem_type="sram",
        size=core_def.size_l1_activation * 8,
        r_bw=core_def.width_l1_activation,
        r_port=0,
        w_port=0,
        rw_port=2,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for weights
    l2_w = MemoryInstance(
        name="l2_w",
        mem_type="sram",
        size=core_def.size_l2_weights * 8,
        r_bw=core_def.width_l2_weigths,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for inputs and outputs
    l2_io = MemoryInstance(
        name="l2_io",
        mem_type="sram",
        size=core_def.size_l2_activation * 8,
        r_bw=core_def.width_l2_activation,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=1,
        auto_cost_extraction=True,
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
        memory_instance=rf_1B,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    # Register file for weight
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_1B,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    # Register file for output
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_2B,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
        ),
        served_dimensions={(0, 1, 0), (0, 0, 1)},
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
        memory_instance=l1_io,
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
    # Second SRAM for weights
    memory_hierarchy_graph.add_memory(
        memory_instance=l2_w,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    # Second SRAM for inputs and output
    memory_hierarchy_graph.add_memory(
        memory_instance=l2_io,
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


def get_memory_hierarchy_OX_K_dataflow(multiplier_array):
    """Memory hierarchy variables"""
    """ size=#bit, bw=#bit"""
    # Defintion of register file for inputs and weights
    rf_1B = MemoryInstance(
        name="rf_1B",
        mem_type="rf",
        size=core_def.size_rf_weight_input * 8,
        r_bw=core_def.width_rf_weight_input,
        r_port=1,
        w_port=1,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of rRegister file for outputs
    rf_2B = MemoryInstance(
        name="rf_4B",
        mem_type="rf",
        size=core_def.size_rf_outputs * 8,
        r_bw=core_def.width_rf_outputs,
        r_port=2,
        w_port=2,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for weights
    l1_w = MemoryInstance(
        name="l1_w",
        mem_type="sram",
        size=core_def.size_l1_weights * 8,
        r_bw=core_def.width_l1_weights,
        r_port=1,
        w_port=1,
        rw_port=0,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for inputs and outputs
    l1_io = MemoryInstance(
        name="l1_io",
        mem_type="sram",
        size=core_def.size_l1_activation * 8,
        r_bw=core_def.width_l1_activation,
        r_port=0,
        w_port=0,
        rw_port=2,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for weights
    l2_w = MemoryInstance(
        name="l2_w",
        mem_type="sram",
        size=core_def.size_l2_weights * 8,
        r_bw=core_def.width_l2_weigths,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        auto_cost_extraction=True,
    )
    # Defintion of first SRAM for inputs and outputs
    l2_io = MemoryInstance(
        name="l2_io",
        mem_type="sram",
        size=core_def.size_l2_activation * 8,
        r_bw=core_def.width_l2_activation,
        r_port=0,
        w_port=0,
        rw_port=2,
        latency=1,
        auto_cost_extraction=True,
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
        memory_instance=rf_1B,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 1)},
    )
    # Register file for weight
    memory_hierarchy_graph.add_memory(
        memory_instance=rf_1B,
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
        memory_instance=l1_io,
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
    # Second SRAM for weights
    memory_hierarchy_graph.add_memory(
        memory_instance=l2_w,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions="all",
    )
    # Second SRAM for inputs and output
    memory_hierarchy_graph.add_memory(
        memory_instance=l2_io,
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
