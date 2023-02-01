from stream.inputs.examples.hardware.cores.Eyeriss_like import get_core as get_eyeriss_like_core
from stream.inputs.examples.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.examples.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.examples.hardware.cores.offchip import get_offchip_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_eyeriss_like_core(id) for id in range(4)]  # 4 identical cores
pooling_core = get_pooling_core(id=4)
simd_core = get_simd_core(id=5)
offchip_core_id = 6
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, 2, 2, 32, 0, pooling_core, simd_core, offchip_core)

global_buffer = None
accelerator = Accelerator("Eyeriss-like-quad-core", cores_graph, global_buffer, offchip_core_id=offchip_core_id)
