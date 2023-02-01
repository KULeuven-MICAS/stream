from stream.inputs.testing.hardware.cores.testing_core1 import get_core as get_testing_core1
from stream.inputs.testing.hardware.cores.testing_core2 import get_core as get_testing_core2
from stream.inputs.examples.hardware.cores.offchip_dual_port import get_offchip_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_testing_core1(id) for id in range(4)]  # 4 cores
offchip_core_id = 4
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, nb_rows=2, nb_cols=2, bandwidth=64, unit_energy_cost=0, offchip_core=offchip_core)

global_buffer = None
accelerator = Accelerator("Testing-4-core-with-offchip", cores_graph, global_buffer, offchip_core_id=offchip_core_id)
