from dataclasses import dataclass, field

from xdsl.context import Context

from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.compiler.kernels.binding import Bindings


@dataclass
class AIEContext(Context):
    registered_kernels: dict[str, AIEKernel] = field(default_factory=dict)
    bindings: Bindings = field(default_factory=lambda: Bindings("npu2"))
