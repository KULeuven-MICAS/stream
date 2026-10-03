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

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from stream.hardware.ports import ANY_OPERAND, READ, WRITE, PortSpec


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
    dma_mm2s: int = 0
    dma_s2mm: int = 0
    dma_channel_bits: int = 0
    dma_buffer_descriptors: int = 0
    dma_iterations: int = 0
    core_id: int = field(default=-1, compare=False)

    #: Core attributes this backend adds to the core's IR.
    core_ir_fields: ClassVar[tuple[str, ...]] = ("max_object_fifo_depth",)

    @classmethod
    def from_core_data(
        cls, core_data: dict[str, Any], core_id: int, shared_mem_group_id: int | None
    ) -> AIE2CoreBackend:
        """The backend of a validated ``aie2`` core description; tiles share no memory, so a tile's DMA ports
        are its own."""
        del shared_mem_group_id
        mem, dma = core_data["memory"], core_data.get("dma", {})
        return cls(
            memory_capacity_bits=mem["capacity"],
            bandwidth_min=mem.get("bandwidth_min", 0),
            bandwidth_max=mem.get("bandwidth_max", 0),
            dma_mm2s=dma.get("mm2s", 0),
            dma_s2mm=dma.get("s2mm", 0),
            dma_channel_bits=dma.get("channel_bits", 0),
            dma_buffer_descriptors=dma.get("buffer_descriptors", 0),
            dma_iterations=dma.get("iterations", 0),
            core_id=core_id,
        )

    def same_hardware(self, other: object) -> bool:
        return self == other

    def has_same_performance(self, other: object) -> bool:
        return self == other

    # ------------------------------------------------------------------
    # Backend protocol — same interface as ZigZagCoreBackend
    # ------------------------------------------------------------------

    def get_memory_capacity(self) -> int:
        """Total top-level memory capacity in bits."""
        return self.memory_capacity_bits

    def get_max_memory_bandwidth(self, type: Literal["read"] | Literal["write"]) -> int:
        """Memory bandwidth in bits/cycle."""
        return self.bandwidth_max

    def memory_ports(self) -> tuple[PortSpec, ...]:
        """The tile DMA: MM2S channels read the tile memory onto streams, S2MM channels write into it."""
        if not self.dma_channel_bits:
            return ()
        return tuple(
            PortSpec(
                self.core_id,
                "dma",
                name,
                channels * self.dma_channel_bits,
                frozenset({(ANY_OPERAND, direction)}),
            )
            for name, channels, direction in (("mm2s", self.dma_mm2s, READ), ("s2mm", self.dma_s2mm, WRITE))
        )

    def get_ir(self) -> dict:
        """Serialize backend-specific fields for the IR dict."""
        return {
            "memory": {
                "capacity_bits": self.memory_capacity_bits,
                "bandwidth_min": self.bandwidth_min,
                "bandwidth_max": self.bandwidth_max,
            },
            "dma": {
                "mm2s": self.dma_mm2s,
                "s2mm": self.dma_s2mm,
                "channel_bits": self.dma_channel_bits,
                "buffer_descriptors": self.dma_buffer_descriptors,
                "iterations": self.dma_iterations,
            },
        }
