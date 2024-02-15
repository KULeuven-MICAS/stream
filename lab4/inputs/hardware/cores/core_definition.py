# This document is used to define common parameters across the three different core architectures
###############################################################################################
#
# Multiplier array sizes
quad_core_multiplier_array_size_2D = [32, 32]
quad_core_multiplier_array_size_3D = [64, 4, 4]
#
#
###############################################################################################
#
# In the following the size and the read/write bus width of the SRAMs in the architecutres
# are defined. The unit for the memory sizes are bytes and for the bus width it is bits.
#
# Size L2 SRAM weight:
size_l2_weights = 1048576 * 3  # 3 MB
# Bus width L2 SRAM weight:
width_l2_weigths = 256
# Size L2 SRAM activation:
size_l2_activation = 1048576  # 1 MB
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
# architectures are defined. The unit for the memory sizes are bytes and for the bus width it is bits.
#
# Size register file weights and inputs:
size_rf_weight_input = 4
# Bus width register file weights and inputs:
width_rf_weight_input = size_rf_weight_input * 8
# Size register file outputs:
size_rf_outputs = 4
# Bus width register file outputs:
width_rf_outputs = size_rf_outputs * 8
#
###############################################################################################
#
# In the following the costs for off chip memory access and the available bus width is defined.
#
size_offchip = 1073741824  # in bytes
width_offchip = 64  # in bits
#
###############################################################################################
#
# The bandwidth of communication links between cores.
inter_core_bandwidth = 64
# The unit energy cost per bit of the communication links between cores.
inter_core_energy_cost = 10
#
###############################################################################################
#
# The operand in the architecutres have the following bit precision:
operand_precision = 8
# One MAC operation has the following costs (in pJ):
energy_mac_operation = 1.0
#
###############################################################################################
#
