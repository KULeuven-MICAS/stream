from stream.inputs.testing.hardware.cores.testing_core1 import get_core as get_testing_core1
from stream.inputs.testing.hardware.cores.testing_core2 import get_core as get_testing_core2
from stream.inputs.examples.hardware.cores.offchip_dual_port import get_offchip_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_testing_core1(0), get_testing_core2(1)]  # 2 cores
offchip_core_id = 2
offchip_core = get_offchip_core(id=offchip_core_id)

# Comment out the offchip_bandwidth because we can get this attribute from the offchip_core (if it is defined), thus no need to manually define it
cores_graph = get_2d_mesh(cores, nb_rows=1, nb_cols=2, bandwidth=64, unit_energy_cost=0, offchip_core=offchip_core) #, offchip_bandwidth=32)

global_buffer = None
accelerator = Accelerator("Testing-2-core-with-offchip", cores_graph, global_buffer, offchip_core_id=offchip_core_id)
