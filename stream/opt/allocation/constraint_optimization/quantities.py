"""Named model expressions the allocation model exposes to objectives, bounds and constraint families."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any


@dataclass
class Quantity:
    """One model expression; ``upper_bound`` is a value the expression provably never exceeds."""

    expr: Any
    upper_bound: float | None = None


class QuantityRegistry:
    """Quantities by name, either scalar or indexed by a hashable key such as a slot or a core."""

    def __init__(self) -> None:
        self._scalars: dict[str, Quantity] = {}
        self._indexed: dict[str, dict[Hashable, Quantity]] = {}

    def add(self, name: str, expr: Any, *, index: Hashable | None = None, upper_bound: float | None = None) -> None:
        quantity = Quantity(expr, upper_bound)
        if index is None:
            if name in self._scalars or name in self._indexed:
                raise ValueError(f"Quantity {name!r} is already registered")
            self._scalars[name] = quantity
            return
        if name in self._scalars:
            raise ValueError(f"Quantity {name!r} is already registered as a scalar")
        entries = self._indexed.setdefault(name, {})
        if index in entries:
            raise ValueError(f"Quantity {name!r}[{index!r}] is already registered")
        entries[index] = quantity

    def get(self, name: str, index: Hashable | None = None) -> Quantity:
        if index is None and name in self._scalars:
            return self._scalars[name]
        if index is not None and index in self._indexed.get(name, {}):
            return self._indexed[name][index]
        label = name if index is None else f"{name}[{index!r}]"
        raise KeyError(f"Unknown quantity {label}; available: {', '.join(self.names())}")

    def indexed(self, name: str) -> dict[Hashable, Quantity]:
        if name not in self._indexed:
            raise KeyError(f"Unknown indexed quantity {name!r}; available: {', '.join(self.names())}")
        return dict(self._indexed[name])

    def names(self) -> list[str]:
        return sorted({*self._scalars, *self._indexed})

    def __contains__(self, name: str) -> bool:
        return name in self._scalars or name in self._indexed
