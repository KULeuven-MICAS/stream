from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.dialects.memref import SubviewOp


@dataclass(frozen=True, repr=False)
class Tensor:
    name: str
    operand_type: FixedBitwidthType
    shape: tuple[int, ...]
    subview: SubviewOp

    def __repr__(self):
        return f"Tensor(name={self.name}, operand_type={self.operand_type}, shape={self.shape})"

    def size_elements(self, shape: tuple[int, ...] | None = None) -> int:
        return prod(shape) if shape is not None else prod(self.shape)

    def size_bits(self, shape: tuple[int, ...] | None = None) -> int:
        return self.operand_type.bitwidth * self.size_elements(shape=shape)
