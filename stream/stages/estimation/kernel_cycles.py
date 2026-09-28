"""Measured per-call kernel costs on AIE silicon."""

from math import prod
from typing import Any

MEASURED_KERNEL_CYCLES: dict[str, float] = {
    "matmul_bf16_bf16_64_64_64": 1730.0,
    "matmul_bf16_bf16_32_32_64": 575.0,
    "matmul_PV": 1536.0,
    "partial_softmax": 4400.0,
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
