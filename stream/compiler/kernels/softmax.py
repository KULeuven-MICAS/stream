from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.arith import AddiOp, ConstantOp, IndexCastOp, MuliOp
from xdsl.dialects.builtin import (
    AnyDenseElement,
    FunctionType,
    MemRefType,
    i32,
)
from xdsl.dialects.func import CallOp
from xdsl.ir import SSAValue
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import (
    ComputationNodeOp,
    StrensorType,
    StrensorVarType,
)
from stream.compiler.kernels.aie_kernel import (
    CONTIGUOUS,
    MAC_ROWS_BFP16,
    AIEKernel,
    R,
    T,
    elementwise_operand_layout,
    induction_variable,
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

    ``causal`` picks the entry point that masks each row past its own position before
    normalizing it, so the tile also has to say where its first row sits globally.
    """

    element_type: AnyDenseElement
    m: int
    n: int
    layout: str
    bfp16_mmul: bool = False
    causal: bool = False

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
        return f"softmax_rows_{'causal_' if self.causal else ''}{self.element_type}"

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
        scalars = [i32, i32, i32] if self.causal else [i32, i32]
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type, op.inputs[1].type, *scalars],
            outputs=[],
        )

    def row_offset(self, op: ComputationNodeOp) -> tuple[Sequence[Operation], SSAValue]:
        """Where the tile's first row sits in the dimension the rows are handed out from.

        The rows are handed out over the cores and over the temporal loop, and a causal
        row masks by its global position, so the kernel needs that position rather than
        the one it holds in the tile. Which core this is is a constant of the spatially
        unrolled node; which iteration it is comes from the loop the row dimension
        drives, named by the ``layer_dim`` its rewrite left on it.
        """
        space = cast(StrensorType, op.output.type).ssis.data
        row_dim = next(iter(space.get_kernel_variables())).dim
        # A point variable carries its position where a sized one carries its size.
        position = {var.dim: var.size for var in (op.spatial_index.data.vars if op.spatial_index else ())}
        ops: list[Operation] = []
        terms: list[SSAValue] = []
        constant, stride = 0, 1
        for var in reversed(space.vars):
            if var.dim != row_dim:
                continue
            if var.type is StrensorVarType.SPATIAL:
                constant += position.get(row_dim, 0) * stride
            elif var.type is StrensorVarType.TEMPORAL:
                ops += [
                    index := IndexCastOp(induction_variable(op, var), i32),
                    size := ConstantOp.from_int_and_width(stride, i32),
                    term := MuliOp(index, size),
                ]
                terms.append(term.result)
            stride *= var.size
        ops.append(offset := ConstantOp.from_int_and_width(constant, i32))
        result = offset.result
        for term in terms:
            ops.append(total := AddiOp(result, term))
            result = total.result
        return ops, result

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        # The only point where the tile the mapping declares meets the one codegen built.
        shape = tuple(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        if shape != (self.m, self.n):
            raise ValueError(f"softmax kernel declares a {self.m} x {self.n} tile but its operand is {shape}")
        ops: list[Operation] = [
            rows := ConstantOp.from_int_and_width(self.m, i32),
            row_len := ConstantOp.from_int_and_width(self.n, i32),
        ]
        arguments: list[SSAValue] = [op.inputs[0], op.inputs[1], rows.result, row_len.result]
        if self.causal:
            offset_ops, offset = self.row_offset(op)
            ops += offset_ops
            arguments.append(offset)
        return [*ops, CallOp(self.function_name, arguments, [])]
