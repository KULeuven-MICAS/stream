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
from stream.compiler.kernels.aie_kernel import AIEKernel, elementwise_operand_layout


@dataclass
class SiluKernel(AIEKernel):
    element_type: AnyDenseElement
    m: int
    n: int
    layout: str

    @property
    def linkwith_name(self) -> str:
        return "silu.o"

    @property
    def function_name(self) -> str:
        return f"silu_{self.element_type}"

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type, op.inputs[1].type, i32],
            outputs=[],
        )

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return [elementwise_operand_layout(self.m, self.n, self.layout) for _ in range(2)]

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        len = prod(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        return [
            len := ConstantOp.from_int_and_width(len, i32),
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], len], []),
        ]
