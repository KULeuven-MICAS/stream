from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation


@dataclass
class CoreCostEntry:
    """Lightweight, backend-agnostic representation of a node-core performance estimate."""

    energy_total: float
    latency_total: float
    ideal_cycle: float
    ideal_temporal_cycle: float
    mem_energy_breakdown: dict[Any, list[float]] = field(default_factory=dict)
    cme: CostModelEvaluation | None = None
    mapping: Any = None
    layer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_cycles_scaled(self, utilization: float) -> float:
        if utilization <= 0:
            return self.latency_total
        return self.latency_total * 100.0 / utilization

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying CME when present to preserve compatibility."""
        if self.cme is not None:
            return getattr(self.cme, name)
        raise AttributeError(name)
