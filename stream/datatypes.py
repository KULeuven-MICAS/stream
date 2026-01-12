from __future__ import annotations

from sympy.core.symbol import Symbol


class LayerDim(Symbol):
    """A Symbol subtype for representing layer dimensions."""

    def __new__(cls, name: str, **assumptions):
        return super().__new__(cls, name, **assumptions)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"LayerDim({self.name!r})"

    def get_idx(self) -> int:
        """Get the integer index of the dimension, assuming the name is of the form 'D{index}'."""
        if self.name.startswith("D") and self.name[1:].isdigit():
            return int(self.name[1:])
        else:
            raise ValueError(f"LayerDim name {self.name!r} is not in the expected format 'D{{index}}'")


InterCoreTiling = tuple[tuple[LayerDim, int], ...]
