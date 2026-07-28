from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
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
from stream.compiler.kernels.aie_kernel import (
    VECTOR_LANES,
    AIEKernel,
    elementwise_operand_layout,
)


@dataclass
class EltwiseMulKernel(AIEKernel):
    element_type: AnyDenseElement
    m: int
    n: int
    layout: str

    @property
    def linkwith_name(self) -> str:
        return "mul.o"

    @property
    def function_name(self) -> str:
        """The vectorized variant where the tile fills whole vectors, else the scalar one.

        The multiply is position independent and all three operands share a layout, so
        the only thing vectorizing asks of the tile is that it leaves no remainder: the
        vectorized kernel steps a whole vector at a time and has no epilogue.
        """
        variant = "vector" if self.m * self.n % VECTOR_LANES == 0 else "scalar"
        return f"eltwise_mul_{self.element_type}_{variant}"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return [elementwise_operand_layout(self.m, self.n, self.layout) for _ in range(3)]

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type, op.inputs[1].type, op.inputs[2].type, i32],
            outputs=[],
        )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        len = prod(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        return [
            len := ConstantOp.from_int_and_width(len, i32),
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], op.inputs[2], len], []),
        ]
