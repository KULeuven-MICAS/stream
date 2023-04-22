# This document is used to defined the hardware architecture for the thesis of Sebastian Karl.
#
# There are three different archtiectures considered: a big single core, a homogeneous quadcore
# and a heterogeneous quadcore
#
###############################################################################################
#
# The multiplier array of the single core is four times as big as the one of the quadcore
#
# Size of one dimension in the 2D multiplier array:
single_core_multiplier_array_size_2D = [64, 64]
quad_core_multiplier_array_size_2D = [32, 32]
single_core_multiplier_array_size_3D = [64, 8, 8]
quad_core_multiplier_array_size_3D = [64, 4, 4]
#
#
###############################################################################################
#
# In the following the size and the read/write bus width of the SRAMs in the architecutres
# are defined. The single core uses directly the size from this file, while the quadcores
# rescale it by a factor of 1/4. The bus width stays the same for all architecutres. The unit
# for the memory sizes are bytes and for the bus width it is bits.
#
# Size L2 SRAM weight:
size_l2_weights = (
    1048576 * 3 * 4
)  # (10*2**20)  # 9437184  # number of elements needed for last resnet18 layer   #  1048576
# Bus width L2 SRAM weight:
width_l2_weigths = 256
# Size L2 SRAM activation:
size_l2_activation = 1048576
# Bus width L2 SRAM activation:
width_l2_activation = 256
# Size L1 SRAM weight:
size_l1_weights = 16384
# Bus width L1 SRAM weight:
width_l1_weights = 128
# Size L1 SRAM activation:
size_l1_activation = 65536
# Bus width L1 SRAM activation:
width_l1_activation = 512
#
###############################################################################################
#
# In the follwoing the size and read/write bus width of the register files in the
# architectures are defined. The size and width is the same for all three architectures. The
# unit for the memory sizes are bytes and for the bus width it is bits.
#
# Size register file weights and inputs:
size_rf_weight_input = 1
# Bus width register file weights and inputs:
width_rf_weight_input = size_rf_weight_input * 8
# Size register file outputs:
size_rf_outputs = 4
# Bus width register file outputs:
width_rf_outputs = size_rf_outputs * 8
#
###############################################################################################
#
# In the following the costs for off chip memory access and the available bus widht is defined.
#
size_offchip = 1073741824  # in bytes
width_offchip = 64
#
###############################################################################################
#
# The core of the quadcore example can communicate over a bus.
#
# This bus has the following bandwidth:
inter_core_bandwidth = 64
#
###############################################################################################
#
# The operand in the architecutres have the following bit precision:
operand_precision = 8
# One MAC operations has the following costs (in pJ):
energy_mac_operation = 1.0
#
###############################################################################################
#
