"""What a design's off-chip traffic costs, when the hardware declares how its DRAM behaves.

One port at one width prices every byte the same. DRAM does not: it charges for leaving a
page, so a transfer walking short runs across rows moves at a fraction of what a
contiguous one does, and the whole array shares one read-plus-write ceiling.

The unit that sets the access pattern DRAM sees is the block a transfer moves at one step,
not one channel's slice of it: the columns streaming a tensor together read adjacent
pieces of the same rows, so DRAM sees their union. That block is the transfer's own tensor,
and its shape against the full off-chip tensor gives the contiguous span.

The allocator uses this twice: a transfer alone moves at no more than one direction's rate,
bits / (contiguous * efficiency), and every off-chip transfer of an iteration together holds
the one memory for bits / (ceiling * efficiency) each, which bounds the step.

`offchip_dram` in the hardware description carries the measurements, in bits per cycle:

    offchip_dram:
      ceiling: 296.5          # read + write together, a contiguous copy both ways
      contiguous: 148.3       # one way, the same copy
      strided:                # one way, strided runs of `span` bytes
        read:  {32: 19.0, 64: 35.2, ...}
        write: {32: 17.1, 64: 28.3, ...}

Efficiency is the strided rate over the contiguous one, log-interpolated in span and taken
as contiguous beyond the widest span measured.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DramProfile:
    ceiling: float
    contiguous: float
    strided: Mapping[str, Mapping[int, float]]

    @classmethod
    def from_description(cls, data: Mapping[str, Any] | None) -> DramProfile | None:
        if not data:
            return None
        strided = {side: {int(k): float(v) for k, v in table.items()} for side, table in data["strided"].items()}
        return cls(ceiling=float(data["ceiling"]), contiguous=float(data["contiguous"]), strided=strided)

    def efficiency(self, span_bytes: float, side: str) -> float:
        table = sorted(self.strided[side].items())
        if span_bytes >= 2 * table[-1][0]:
            return 1.0
        points = [(math.log(s), bw / self.contiguous) for s, bw in table] + [(math.log(2 * table[-1][0]), 1.0)]
        x = math.log(max(span_bytes, table[0][0]))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x <= x1:
                return min(1.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))
        return 1.0

    def cycles(self, transfers: Iterable[tuple[float, float, str]]) -> float:
        """Cycles of the shared DRAM that (bits, span_bytes, side) transfers occupy."""
        return sum(bits / (self.ceiling * self.efficiency(span, side)) for bits, span, side in transfers)


def contiguous_span_bytes(block: tuple[int, ...], full: tuple[int, ...] | None, element_bits: int) -> float:
    """Bytes the block covers contiguously in the row-major full tensor: its innermost
    extent, carried outward for as long as the dimensions inside are whole."""
    if not block:
        return 0.0
    if not full or len(full) != len(block):
        return block[-1] * element_bits / 8
    span = 1
    for size, whole in zip(reversed(block), reversed(full)):
        span *= size
        if size != whole:
            break
    return span * element_bits / 8


__all__ = ["DramProfile", "contiguous_span_bytes"]
