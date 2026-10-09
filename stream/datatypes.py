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

ELEMENT_BITS: dict[str, int] = {
    "int4": 4,
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "bf16": 16,
    "fp16": 16,
    "int32": 32,
    "fp32": 32,
}
"""Bits of each element type a core's ``operand_precision`` can name."""
