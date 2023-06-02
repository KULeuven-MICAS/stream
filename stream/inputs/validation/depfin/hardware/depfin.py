from stream.inputs.validation.depfin.hardware.cores.digital import (
    get_core as get_digital_core,
)

from stream.inputs.validation.depfin.hardware.cores.offchip import (
    get_core as get_offchip_core,
)
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_digital_core(0)]  # 1 core
offchip_core_id = 1
offchip_core = get_offchip_core(id=offchip_core_id)

# Comment out the offchip_bandwidth because we can get this attribute from the offchip_core (if it is defined), thus no need to manually define it
cores_graph = get_2d_mesh(
    cores,
    nb_rows=1,
    nb_cols=1,
    bandwidth=64,
    unit_energy_cost=0,
    offchip_core=offchip_core,
)

accelerator = Accelerator("depfin", cores_graph, offchip_core_id=offchip_core_id)
