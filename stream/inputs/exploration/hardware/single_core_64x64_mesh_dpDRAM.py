from stream.inputs.exploration.hardware.cores.C64_K64 import get_core
from stream.inputs.exploration.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.exploration.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.exploration.hardware.cores.offchip_1r1w import get_offchip_core
from stream.inputs.exploration.hardware.nocs.mesh_2d import get_2d_mesh
from stream.inputs.exploration.hardware.nocs.comm_bus import get_comm_bus
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_core(id) for id in range(1)]
pooling_core = get_pooling_core(id=1)
simd_core = get_simd_core(id=2)
offchip_core_id = 3
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, 1, 1, 64, 0, pooling_core, simd_core, offchip_core)
# cores_graph = get_comm_bus(cores, 64, 0, pooling_core, simd_core, offchip_core)

accelerator = Accelerator(
    "single_core_64x64_mesh_dpDRAM", cores_graph, offchip_core_id=offchip_core_id
)
