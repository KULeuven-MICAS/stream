from dataclasses import dataclass, field

from xdsl.context import Context

from stream.compiler.kernels.aie_kernel import AIEKernel


@dataclass
class AIEContext(Context):
    registered_kernels: dict[str, AIEKernel] = field(default_factory=dict)
