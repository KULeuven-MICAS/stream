from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import FixedBitwidthType, MemRefType
from xdsl.dialects.memref import AllocOp, SubviewOp


@dataclass(frozen=True, repr=False)
class Tensor:
    name: str
    operand_type: FixedBitwidthType
    shape: tuple[int, ...]
    subview: SubviewOp

    def __repr__(self):
        return f"Tensor(name={self.name}, operand_type={self.operand_type}, shape={self.shape})"

    @classmethod
    def create(cls, name: str, operand_type: FixedBitwidthType, shape: tuple[int, ...]) -> "Tensor":
        """Build a Tensor with a fresh full-extent memref subview (the canonical construction)."""
        memref_type = MemRefType(operand_type, shape)
        source = AllocOp([], [], memref_type)
        subview = SubviewOp.from_static_parameters(
            source=source,
            source_type=memref_type,
            offsets=[0] * len(shape),
            sizes=list(shape),
            strides=[1] * len(shape),
        )
        return cls(name=name, operand_type=operand_type, shape=shape, subview=subview)

    def size_elements(self, shape: tuple[int, ...] | None = None) -> int:
        return prod(shape) if shape is not None else prod(self.shape)

    def size_bits(self, shape: tuple[int, ...] | None = None) -> int:
        return self.operand_type.bitwidth * self.size_elements(shape=shape)
