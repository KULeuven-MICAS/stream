from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir.affine import AffineDimExpr


@dataclass(frozen=True, repr=False)
class LayerDim(AffineDimExpr):
    prefix: str = "z"

    def __str__(self) -> str:
        return f"{self.prefix}{self.position}"

    def __repr__(self) -> str:
        return str(self)


InterCoreTiling = tuple[tuple[LayerDim, int], ...]
