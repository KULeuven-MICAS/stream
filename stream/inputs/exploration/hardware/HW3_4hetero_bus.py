from stream.inputs.exploration.hardware.cores.C16_K16 import get_core as get_core_C16_K16
from stream.inputs.exploration.hardware.cores.OX16_K16 import get_core as get_core_OX16_K16
from stream.inputs.exploration.hardware.cores.OX28_FX3_FY3 import get_core as get_core_OX28_FX3_FY3
from stream.inputs.exploration.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.exploration.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.exploration.hardware.cores.offchip_1rw import get_offchip_core
from stream.inputs.exploration.hardware.nocs.mesh_2d import get_2d_mesh
from stream.inputs.exploration.hardware.nocs.comm_bus import get_comm_bus
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_core_C16_K16(id) for id in range(2)]  # 2 identical cores
cores.append(get_core_OX16_K16(id=2))
cores.append(get_core_OX28_FX3_FY3(id=3))
pooling_core = get_pooling_core(id=4)
simd_core = get_simd_core(id=5)
offchip_core_id = 6
offchip_core = get_offchip_core(id=offchip_core_id)

# cores_graph = get_2d_mesh(cores, 2, 2, 64, 0, pooling_core, simd_core, offchip_core)
cores_graph = get_comm_bus(cores, 64, 0, pooling_core, simd_core, offchip_core)

accelerator = Accelerator(
    "HW3_4hetero_bus", cores_graph, offchip_core_id=offchip_core_id
)
