"""Per-call kernel costs, as the attached kernel library measured them.

An anchor is the median span of one call of a symbol, which a hardware trace pairs from
its own event0/event1 markers, so recalibrating is rerunning the trace and editing the
library's manifest. Scaling to a node's per-iteration latency is linear in operation count
from the anchor call.

The mapping-supplied ``utilization`` percentage remains the fallback for a symbol the
library gives no anchor for; where an anchor exists it wins, because it is a measurement
of the deployed binary rather than a hand-fed estimate.
"""

from math import prod
from typing import Any

from stream.compiler.kernels import manifest
from stream.compiler.kernels.manifest import CALL_DIMS


def anchor(kernel: Any) -> float | tuple[float, int] | None:
    """This kernel's measured per-call cost, as the kernel library declared it.

    A shape the library measured wins; otherwise its per-operation anchor stands in, which
    is what keeps a shape nobody timed priced from silicon rather than a hand-fed guess."""
    key = getattr(kernel, "manifest_key", None)
    shape = kernel.call_shape() if hasattr(kernel, "call_shape") else {}
    return manifest.cycles(key, shape)


def calls_ops(kernel: Any) -> int:
    """The operations one call of this kernel covers, from its own tile dimensions."""
    return prod(int(getattr(kernel, d)) for d in CALL_DIMS if getattr(kernel, d, None) is not None)


def measured_latency(kernel: Any, ops: int) -> float | None:
    """Cycles for ``ops`` operations of this kernel, when its symbol has a measured anchor."""
    name = getattr(kernel, "function_name", None)
    if name is None:
        return None
    entry = anchor(kernel)
    if entry is None:
        return None
    cycles = entry
    if isinstance(entry, tuple):
        cycles, anchor_ops = entry
        return cycles * ops / anchor_ops
    per_call = calls_ops(kernel)
    if per_call <= 0:
        return None
    return cycles * ops / per_call
