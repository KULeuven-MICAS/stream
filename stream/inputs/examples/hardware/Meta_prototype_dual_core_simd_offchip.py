from stream.inputs.examples.hardware.cores.Meta_prototype_like import get_core as get_meta_prototype_core
from stream.inputs.examples.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.examples.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.examples.hardware.cores.offchip import get_offchip_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_meta_prototype_core(id) for id in range(2)]  # 2 identical cores
pooling_core = get_pooling_core(id=2)
simd_core = get_simd_core(id=3)
offchip_core_id = 4
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, 1, 2, 64, 0, pooling_core, simd_core, offchip_core)

global_buffer = None
accelerator = Accelerator("Meta-proto-2-core-with-pooling-and-offchip-cores", cores_graph, global_buffer, offchip_core_id=offchip_core_id)
