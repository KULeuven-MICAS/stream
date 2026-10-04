"""Core backends: the hardware details a :class:`~stream.hardware.architecture.core.Core` delegates to. Each
implements ``get_memory_capacity``, ``get_max_memory_bandwidth``, ``get_ir`` and ``memory_ports``, the top-level
memory ports as :class:`~stream.hardware.ports.PortSpec` (empty when the backend models none)."""

from stream.hardware.architecture.backends.aie2 import AIE2CoreBackend
from stream.hardware.architecture.backends.zigzag import ZigZagCoreBackend

#: Union of all supported backend types.
#: Extend this when adding a new backend.
AnyBackend = ZigZagCoreBackend | AIE2CoreBackend

__all__ = [
    "AIE2CoreBackend",
    "AnyBackend",
    "ZigZagCoreBackend",
]
