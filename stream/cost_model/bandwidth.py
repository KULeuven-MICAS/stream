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
    direction at a contiguous access, and ``strided[direction]`` per contiguous span in bytes, as measured. Where
    nothing is measured, ``burst`` is the bytes the memory moves per access, of which a shorter contiguous run uses
    only its part."""

    ceiling: float
    contiguous: float
    strided: Mapping[str, Mapping[int, float]]
    burst: int | None = None

    @classmethod
    def from_description(cls, data: Mapping[str, Any]) -> BandwidthModel:
        measured = data.get("strided") or {}
        strided = {side: {int(k): float(v) for k, v in measured.get(side, {}).items()} for side in DIRECTIONS}
        burst = data.get("burst")
        return cls(ceiling=float(data["ceiling"]), contiguous=float(data["contiguous"]), strided=strided, burst=burst)

    @classmethod
    def flat(cls, bits_per_cycle: float) -> BandwidthModel:
        """A rate that every access pattern reaches, as a hardware description's port width declares."""
        return cls(ceiling=bits_per_cycle, contiguous=bits_per_cycle, strided={side: {} for side in DIRECTIONS})

    def efficiency(self, span_bytes: float, direction: str) -> float:
        """Fraction of the contiguous rate a transfer gets whose contiguous span is ``span_bytes``."""
        table = sorted(self.strided[direction].items())
        if not table and self.burst:
            return span_bytes / (math.ceil(span_bytes / self.burst) * self.burst)
        if not table or span_bytes >= 2 * table[-1][0]:
            return 1.0
        points = [(math.log(s), bw / self.contiguous) for s, bw in table] + [(math.log(2 * table[-1][0]), 1.0)]
        x = math.log(max(span_bytes, table[0][0]))
        for (x0, y0), (x1, y1) in pairwise(points):
            if x <= x1:
                return min(1.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))
        return 1.0
