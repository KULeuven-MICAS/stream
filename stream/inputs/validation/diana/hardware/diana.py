from stream.inputs.validation.diana.hardware.cores.digital import (
    get_core as get_digital_core,
)
from stream.inputs.validation.diana.hardware.cores.analog import (
    get_core as get_analog_core,
)
from stream.inputs.validation.diana.hardware.cores.pooling import (
    get_core as get_pooling_core,
)
from stream.inputs.validation.diana.hardware.cores.simd import get_core as get_simd_core

from stream.inputs.validation.diana.hardware.cores.offchip_dual_port import (
    get_core as get_offchip_core,
)
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [
    get_digital_core(0),
    get_analog_core(1),
    get_pooling_core(2),
    get_simd_core(3),
]  # 4 cores
offchip_core_id = 4
offchip_core = get_offchip_core(id=offchip_core_id)

# Comment out the offchip_bandwidth because we can get this attribute from the offchip_core (if it is defined), thus no need to manually define it
cores_graph = get_2d_mesh(
    cores,
    nb_rows=2,
    nb_cols=2,
    bandwidth=64,
    unit_energy_cost=0,
    offchip_core=offchip_core,
)

accelerator = Accelerator("diana", cores_graph, offchip_core_id=offchip_core_id)
