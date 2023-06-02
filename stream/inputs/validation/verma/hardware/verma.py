from stream.inputs.validation.verma.hardware.cores.core_0 import (
    get_core as get_core_0,
)
from stream.inputs.validation.verma.hardware.cores.core_1 import (
    get_core as get_core_1,
)
from stream.inputs.validation.verma.hardware.cores.core_2 import (
    get_core as get_core_2,
)
from stream.inputs.validation.verma.hardware.cores.core_3 import (
    get_core as get_core_3,
)
from stream.inputs.validation.verma.hardware.cores.core_4 import (
    get_core as get_core_4,
)
from stream.inputs.validation.verma.hardware.cores.offchip_dual_port import (
    get_core as get_offchip_core,
)

from stream.inputs.validation.verma.hardware.noc.noc import shared_memories_with_offchip
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [
    get_core_0(0),
    get_core_1(1),
    get_core_2(2),
    get_core_3(3),
    get_core_4(4),
]  # 5 cores
offchip_core_id = 5
offchip_core = get_offchip_core(id=offchip_core_id)

# Comment out the offchip_bandwidth because we can get this attribute from the offchip_core (if it is defined), thus no need to manually define it
cores_graph = shared_memories_with_offchip(
    cores,
    unit_energy_cost=0,
    offchip_core=offchip_core,
)

accelerator = Accelerator("verma", cores_graph, offchip_core_id=offchip_core_id)
