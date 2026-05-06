"""AIE2 core backend — lightweight tile description for the simplified cost model.

Unlike ZigZag-backed cores, AIE2 tiles do **not** carry a full operational-array
+ multi-level memory-hierarchy model.  They expose only the information
required by the Stream scheduler and simplified AIE cost estimator:

* ``memory_capacity_bits`` — total usable memory on the tile (in bits).
* ``bandwidth_min`` / ``bandwidth_max`` — memory bandwidth in bits/cycle.

This keeps the YAML definition minimal and avoids dragging in ZigZag
concepts that do not apply to the AIE2 architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AIE2CoreBackend:
    """Backend for AIE2 tiles.

    Parameters
    ----------
    memory_capacity_bits:
        Total top-level memory capacity of the tile in **bits**.
    bandwidth_min:
        Minimum memory bandwidth in **bits/cycle**.
    bandwidth_max:
        Maximum memory bandwidth in **bits/cycle**.
    """

    memory_capacity_bits: int
    bandwidth_min: int = 0
    bandwidth_max: int = 0

    # ------------------------------------------------------------------
    # Backend protocol — same interface as ZigZagCoreBackend
    # ------------------------------------------------------------------

    def get_memory_capacity(self) -> int:
        """Total top-level memory capacity in bits."""
        return self.memory_capacity_bits

    def get_max_memory_bandwidth(self, type: Literal["read"] | Literal["write"]) -> int:
        """Memory bandwidth in bits/cycle."""
        return self.bandwidth_max

    def get_ir(self) -> dict:
        """Serialize backend-specific fields for the IR dict."""
        return {
            "memory": {
                "capacity_bits": self.memory_capacity_bits,
                "bandwidth_min": self.bandwidth_min,
                "bandwidth_max": self.bandwidth_max,
            },
        }
