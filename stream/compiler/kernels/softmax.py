from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from snaxc.ir.tsl import TiledStridedLayout
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AnyDenseElement,
    FunctionType,
    MemRefType,
    i32,
)
from xdsl.dialects.func import CallOp
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import CONTIGUOUS, AIEKernel, elementwise_operand_layout

SOFTMAX_VECTOR_LANES = 64
"""Elements softmax.cc reduces per step; it has no epilogue, so a shorter tail is dropped."""


@dataclass
class SoftmaxKernel(AIEKernel):
    """One call of softmax.cc normalizes its whole buffer, so the tile is one row.

    The kernel keeps a single scalar maximum and a single scalar sum over the length it
    is given, and takes no stride, so handing it anything but exactly one complete row
    silently normalizes across rows instead of along them.
    """

    element_type: AnyDenseElement
    m: int
    n: int
    layout: str

    def __post_init__(self) -> None:
        if self.m != 1:
            raise ValueError(f"softmax reduces a whole row per call, so its tile is 1 x n, not {self.m} x {self.n}")
        if self.layout != CONTIGUOUS:
            raise ValueError(f"softmax reads its row linearly and needs the {CONTIGUOUS!r} layout, not {self.layout!r}")
        if self.n % SOFTMAX_VECTOR_LANES:
            raise ValueError(
                f"softmax drops the tail of a row that is not a multiple of {SOFTMAX_VECTOR_LANES}: {self.n}"
            )

    @property
    def unique_name(self) -> str:
        return f"{self.function_name}_{self.m}_{self.n}_{self.layout}"

    @property
    def linkwith_name(self) -> str:
        return "softmax.o"

    @property
    def function_name(self) -> str:
        return f"softmax_{self.element_type}"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return [elementwise_operand_layout(self.m, self.n, self.layout) for _ in range(2)]

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type, op.inputs[1].type, i32],
            outputs=[],
        )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        # The only point where the tile the mapping declares meets the one codegen built.
        shape = tuple(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        if shape != (self.m, self.n):
            raise ValueError(f"softmax kernel declares a {self.m} x {self.n} tile but its operand is {shape}")
        return [
            row_len := ConstantOp.from_int_and_width(self.n, i32),
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], row_len], []),
        ]
