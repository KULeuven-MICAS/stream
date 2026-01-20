from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir.affine import AffineDimExpr


@dataclass(frozen=True, repr=False)
class LayerDim(AffineDimExpr):
    def __str__(self) -> str:
        return f"z{self.position}"

    def __repr__(self) -> str:
        return str(self)


InterCoreTiling = tuple[tuple[LayerDim, int], ...]
