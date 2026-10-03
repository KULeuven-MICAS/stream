"""Core backends: the hardware details a :class:`~stream.hardware.architecture.core.Core` delegates to, built
per core-type namespace by :data:`BACKEND_BUILDERS`. Each backend implements the methods ``Core`` calls on it and
``memory_ports``, its top-level memory ports as :class:`~stream.hardware.ports.PortSpec` (empty when none)."""

from collections.abc import Callable
from typing import Any

from stream.hardware.architecture.backends.aie2 import AIE2CoreBackend
from stream.hardware.architecture.backends.zigzag import ZigZagCoreBackend

#: Union of all supported backend types.
#: Extend this when adding a new backend.
AnyBackend = ZigZagCoreBackend | AIE2CoreBackend

#: Core-type namespace to the builder of its backend from (core data, core id, shared memory group id).
BACKEND_BUILDERS: dict[str, Callable[[dict[str, Any], int, int | None], AnyBackend]] = {
    "aie2": AIE2CoreBackend.from_core_data,
    "zigzag": ZigZagCoreBackend.from_core_data,
}

__all__ = [
    "BACKEND_BUILDERS",
    "AIE2CoreBackend",
    "AnyBackend",
    "ZigZagCoreBackend",
]
