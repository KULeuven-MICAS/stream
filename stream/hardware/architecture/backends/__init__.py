"""Pluggable core backends for Stream.

Each backend supplies the hardware-specific details that a :class:`~stream.hardware.architecture.core.Core`
delegates to.  The base :class:`Core` is backend-agnostic; it only holds identity and
stream-level scheduling attributes.

Both backends implement the same protocol:
- ``get_memory_capacity() -> int``
- ``get_max_memory_bandwidth(type) -> int``
- ``get_ir() -> dict``

Available backends
------------------

``ZigZagCoreBackend``
    Thin wrapper around :mod:`zigzag.hardware.architecture.accelerator.Accelerator`
    — full operational-array + memory-hierarchy model used by the ZigZag cost
    estimator.

``AIE2CoreBackend``
    Lightweight description for AIE2 tiles carrying memory capacity and
    bandwidth for the simplified cost model.
"""

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
