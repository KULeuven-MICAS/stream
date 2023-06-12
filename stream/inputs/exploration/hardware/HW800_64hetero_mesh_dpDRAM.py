from stream.inputs.exploration.hardware.cores.C16_K16 import get_core as get_core_C16_K16
from stream.inputs.exploration.hardware.cores.OX16_K16 import get_core as get_core_OX16_K16
from stream.inputs.exploration.hardware.cores.OX28_FX3_FY3 import get_core as get_core_OX28_FX3_FY3
from stream.inputs.exploration.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.exploration.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.exploration.hardware.cores.offchip_1r1w import get_offchip_core
from stream.inputs.exploration.hardware.nocs.mesh_2d import get_2d_mesh
from stream.inputs.exploration.hardware.nocs.comm_bus import get_comm_bus
from stream.classes.hardware.architecture.accelerator import Accelerator

def my_core_pattern(i):
    return [get_core_C16_K16(4*i), get_core_C16_K16(4*i + 1), get_core_OX16_K16(4*i + 2), get_core_OX28_FX3_FY3(4*i + 3)]

cores = []
core_pattern_repetitive_time = 16
for i in range(core_pattern_repetitive_time):
    cores += my_core_pattern(i)

pooling_core = get_pooling_core(id=64)
simd_core = get_simd_core(id=65)
offchip_core_id = 66
offchip_core = get_offchip_core(id=offchip_core_id)

# cores_graph = get_2d_mesh(cores, 2, 2, 64, 0, pooling_core, simd_core, offchip_core)
cores_graph = get_comm_bus(cores, 64, 0, pooling_core, simd_core, offchip_core)

accelerator = Accelerator(
    "HW800_64hetero_mesh_dpDRAM", cores_graph, offchip_core_id=offchip_core_id
)
