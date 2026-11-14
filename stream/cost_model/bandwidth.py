"""The measured bandwidth of a core that every transfer through it shares, such as off-chip memory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

DIRECTIONS = ("read", "write")


@dataclass(frozen=True)
class BandwidthModel:
    """Rates in bits per cycle: ``ceiling`` for reads and writes together, ``contiguous`` for one
    direction at a contiguous access, and ``strided[direction]`` per contiguous span in bytes, as measured, and
    ``adjacent[direction]`` per span where neighbouring channels move neighbouring spans of the same rows. Where
    nothing is measured, ``burst`` is the bytes the memory moves per access, of which a shorter contiguous run uses
    only its part."""

    ceiling: float
    contiguous: float
    strided: Mapping[str, Mapping[int, float]]
    burst: int | None = None
    adjacent: Mapping[str, Mapping[int, float]] = field(default_factory=dict)

    @classmethod
    def from_description(cls, data: Mapping[str, Any]) -> BandwidthModel:
        strided, adjacent = (
            {side: {int(k): float(v) for k, v in (data.get(key) or {}).get(side, {}).items()} for side in DIRECTIONS}
            for key in ("strided", "adjacent")
        )
        burst = data.get("burst")
        return cls(
            ceiling=float(data["ceiling"]),
            contiguous=float(data["contiguous"]),
            strided=strided,
            burst=burst,
            adjacent=adjacent,
        )

    @classmethod
    def flat(cls, bits_per_cycle: float) -> BandwidthModel:
        """A rate that every access pattern reaches, as a hardware description's port width declares."""
        return cls(ceiling=bits_per_cycle, contiguous=bits_per_cycle, strided={side: {} for side in DIRECTIONS})

    def scaled(self, factor: float) -> BandwidthModel:
        """This model with every rate multiplied by ``factor``, as for identical ports used in parallel."""
        return BandwidthModel(
            ceiling=self.ceiling * factor,
            contiguous=self.contiguous * factor,
            strided={
                side: {span: rate * factor for span, rate in table.items()} for side, table in self.strided.items()
            },
            burst=self.burst,
            adjacent={
                side: {span: rate * factor for span, rate in table.items()} for side, table in self.adjacent.items()
            },
        )

    def full_rate_span(self, direction: str) -> float:
        """The shortest contiguous span, in bytes, at which a transfer in ``direction`` gets the contiguous rate."""
        table = sorted(self.strided[direction].items())
        if table:
            return 2 * table[-1][0]
        return self.burst or 0.0

    def efficiency(self, span_bytes: float, direction: str, run_bytes: float | None = None) -> float:
        """Fraction of the contiguous rate a transfer gets whose contiguous span is ``span_bytes``; where neighbouring
        channels move it in adjacent runs of ``run_bytes``, at most what such runs get."""
        table = sorted(self.strided[direction].items())
        if not table and self.burst:
            return span_bytes / (math.ceil(span_bytes / self.burst) * self.burst)
        whole = self._interpolated(table, span_bytes)
        adjacent = sorted(self.adjacent.get(direction, {}).items())
        if run_bytes is None or run_bytes >= span_bytes or not adjacent:
            return whole
        return min(whole, self._interpolated(adjacent, run_bytes))

    def _interpolated(self, table: list[tuple[int, float]], span_bytes: float) -> float:
        """The fraction of the contiguous rate ``table`` gives a span, log-linear between its spans and reaching the
        contiguous rate at twice its longest."""
        if not table or span_bytes >= 2 * table[-1][0]:
            return 1.0
        points = [(math.log(s), bw / self.contiguous) for s, bw in table] + [(math.log(2 * table[-1][0]), 1.0)]
        x = math.log(max(span_bytes, table[0][0]))
        for (x0, y0), (x1, y1) in pairwise(points):
            if x <= x1:
                return min(1.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))
        return 1.0
