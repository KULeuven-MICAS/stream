from stream.inputs.aie.hardware.aie_core1 import (
    get_core as aie_core1,
)
from stream.inputs.aie.hardware.aie_core2 import (
    get_core as aie_core2,
)
from stream.inputs.aie.hardware.aie_core2 import (
    get_core as aie_core4,
)
# from stream.inputs.aie.hardware.mem_tile import (
#     get_memTile_core as mem_tile,
# )
from stream.inputs.aie.hardware.shim_dma_core import (
    get_shim_dma_core as shim_core
)
from stream.inputs.examples.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [aie_core1(0),aie_core2(1),aie_core2(2),aie_core4(3)]  # 4 cores
pooling_core = get_pooling_core(id=4)
# mem_tile_core = mem_tile(id=4)
offchip_core_id = 5
offchip_core = shim_core(id=offchip_core_id) # basically DRAM

# Comment out the offchip_bandwidth because we can get this attribute from the offchip_core (if it is defined), thus no need to manually define it
cores_graph = get_2d_mesh(
    cores,
    nb_rows=1,
    nb_cols=4,
    bandwidth=64*8,
    pooling_core=pooling_core,
    unit_energy_cost=0,
    # mem_tile_core=mem_tile_core,
    offchip_core=offchip_core,
)  # , offchip_bandwidth=32)

accelerator = Accelerator(
    "AIE2_IPU", cores_graph, offchip_core_id=offchip_core_id
)
