import lab4.inputs.hardware.cores.core_definition as core_def

from lab4.inputs.hardware.cores.ox_fx_fy import get_core as get_ox_fx_fy_core
from lab4.inputs.hardware.cores.ox_k import get_core as get_ox_k_core
from lab4.inputs.hardware.cores.c_k import get_core as get_c_k_core
from lab4.inputs.hardware.cores.offchip import get_core as get_offchip_core


from stream.inputs.examples.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.examples.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator


# Because it's a quadcore system, we need to rescale the memory sizes
core_def.size_l2_weights = core_def.size_l2_weights / 4
core_def.size_l2_activation = core_def.size_l2_activation / 4
core_def.size_l1_weights = core_def.size_l1_weights / 4
core_def.size_l1_activation = core_def.size_l1_activation / 4

# Get all the computational cores, assigning an id
cores = [
    get_ox_fx_fy_core(id=0, quad_core=True),
    get_ox_k_core(id=1, quad_core=True),
    get_c_k_core(id=2, quad_core=True),
    get_c_k_core(id=3, quad_core=True),
]

pooling_core = get_pooling_core(id=4)
simd_core = get_simd_core(id=5)
offchip_core_id = 6
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, 2, 2, core_def.inter_core_bandwidth, 0, pooling_core, simd_core, offchip_core)

accelerator = Accelerator(
    "heterogeneous-quadcore",
    cores_graph,
    offchip_core_id=offchip_core_id,
)
