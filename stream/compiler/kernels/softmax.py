from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
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
from stream.compiler.kernels.aie_kernel import (
    CONTIGUOUS,
    MAC_ROWS_BFP16,
    AIEKernel,
    R,
    T,
    elementwise_operand_layout,
)

SOFTMAX_VECTOR_LANES = 64
"""Elements softmax.cc reduces per step; it has no epilogue, so a shorter tail is dropped."""


@dataclass
class SoftmaxKernel(AIEKernel):
    """One call of softmax.cc normalizes an m x n tile, one row at a time.

    The kernel keeps a single scalar maximum and a single scalar sum per row and takes
    no stride, so n has to be the whole reduction and the rows have to be contiguous;
    anything narrower normalizes across a fraction of the row instead of along it.
    The row loop lives in the kernel because a core call takes a bare pointer, which
    carries no offset for an MLIR-side view of a single row.
    """

    element_type: AnyDenseElement
    m: int
    n: int
    layout: str
    bfp16_mmul: bool = False

    def __post_init__(self) -> None:
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
        return f"softmax_rows_{self.element_type}"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return [self._row_major() for _ in range(2)]

    def _row_major(self) -> TiledStridedLayout:
        """Row major, spelled over the MAC tile bounds of the GEMM either side of it
        where those divide, since a transform is read off matching tile bounds."""
        rows = MAC_ROWS_BFP16 if self.bfp16_mmul else R
        if self.m % rows or self.n % T:
            return elementwise_operand_layout(self.m, self.n, self.layout, rows)
        return TiledStridedLayout(
            [
                TiledStride([Stride(rows * self.n, self.m // rows), Stride(self.n, rows)]),
                TiledStride([Stride(T, self.n // T), Stride(1, T)]),
            ]
        )

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type, op.inputs[1].type, i32, i32],
            outputs=[],
        )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        # The only point where the tile the mapping declares meets the one codegen built.
        shape = tuple(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        if shape != (self.m, self.n):
            raise ValueError(f"softmax kernel declares a {self.m} x {self.n} tile but its operand is {shape}")
        return [
            rows := ConstantOp.from_int_and_width(self.m, i32),
            row_len := ConstantOp.from_int_and_width(self.n, i32),
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], rows, row_len], []),
        ]
