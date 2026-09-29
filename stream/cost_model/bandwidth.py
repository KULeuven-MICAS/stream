"""The measured bandwidth of a core that every transfer through it shares, such as off-chip memory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

DIRECTIONS = ("read", "write")


@dataclass(frozen=True)
class BandwidthModel:
    """Rates in bits per cycle: ``ceiling`` for reads and writes together, ``contiguous`` for one
    direction at a contiguous access, and ``strided[direction]`` per contiguous span in bytes."""

    ceiling: float
    contiguous: float
    strided: Mapping[str, Mapping[int, float]]

    @classmethod
    def from_description(cls, data: Mapping[str, Any]) -> BandwidthModel:
        strided = {side: {int(k): float(v) for k, v in data["strided"][side].items()} for side in DIRECTIONS}
        return cls(ceiling=float(data["ceiling"]), contiguous=float(data["contiguous"]), strided=strided)

    def efficiency(self, span_bytes: float, direction: str) -> float:
        """Fraction of the contiguous rate a transfer gets whose contiguous span is ``span_bytes``."""
        table = sorted(self.strided[direction].items())
        if span_bytes >= 2 * table[-1][0]:
            return 1.0
        points = [(math.log(s), bw / self.contiguous) for s, bw in table] + [(math.log(2 * table[-1][0]), 1.0)]
        x = math.log(max(span_bytes, table[0][0]))
        for (x0, y0), (x1, y1) in pairwise(points):
            if x <= x1:
                return min(1.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))
        return 1.0


def contiguous_span_bytes(block: tuple[int, ...], full: tuple[int, ...], element_bits: int) -> float:
    """Bytes the block covers contiguously in the row-major full tensor."""
    span = 1
    for size, whole in zip(reversed(block), reversed(full), strict=True):
        span *= size
        if size != whole:
            break
    return span * element_bits / 8
