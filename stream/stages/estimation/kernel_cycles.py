"""Measured per-call kernel costs on AIE silicon.

Keyed by the kernel's linked function symbol -- the same name a hardware trace's
event0/event1 spans pair to -- so an entry here is directly the median span of
one call of that symbol, and recalibrating is rerunning the trace. Scaling to a
node's per-iteration latency is linear in operation count from the anchor call.

The mapping-supplied ``utilization`` percentage remains the fallback for symbols
without an anchor; where an anchor exists it wins, because it is a measurement
of the deployed binary rather than a hand-fed estimate.

Sources: ~/stream-dse-iron-slides/260902_mha_array_sweep (per-kernel-call spans,
seq 2048, 8 heads, column 3) and docs/source/aie_calibration.md.
"""

from math import prod
from typing import Any

MEASURED_KERNEL_CYCLES: dict[str, float] = {
    "matmul_bf16_bf16_64_64_64": 1730.0,
    "matmul_bf16_bf16_32_32_64": 575.0,
    "matmul_PV": 1536.0,
    "partial_softmax": 4400.0,
    # The MAC-tiled handover variant measures 2.09x the row-major body (occupancy trace,
    # 260827_softmax_kernel: 5,149 -> 10,742 cycles per step). This ratio is what makes a
    # wide-softmax row layout -- which forces the tiled handover -- rank truthfully: the
    # extra row halves the calls per core and the tiled body doubles each call back.
    "partial_softmax_mode": 9200.0,
    "matmul_softmax": 6130.0,
}

_CALL_DIMS = ("m", "k", "n")


def calls_ops(kernel: Any) -> int:
    """The operations one call of this kernel covers, from its own tile dimensions."""
    return prod(int(getattr(kernel, d)) for d in _CALL_DIMS if getattr(kernel, d, None) is not None)


def measured_latency(kernel: Any, ops: int) -> float | None:
    """Cycles for ``ops`` operations of this kernel, when its symbol has a measured anchor."""
    name = getattr(kernel, "function_name", None)
    if name is None:
        return None
    cycles = MEASURED_KERNEL_CYCLES.get(name)
    if cycles is None:
        return None
    per_call = calls_ops(kernel)
    if per_call <= 0:
        return None
    return cycles * ops / per_call
