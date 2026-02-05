from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

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
from stream.compiler.kernels.aie_kernel import AIEKernelWithZeroing


@dataclass
class MatVecKernel(AIEKernelWithZeroing):
    element_type: AnyDenseElement

    @property
    def linkwith_name(self) -> str:
        return "mv.o"

    @property
    def zero_name(self) -> str:
        return f"zero_vectorized_{self.element_type}"

    def zero_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(inputs=[op.output.type], outputs=[])

    @property
    def function_name(self) -> str:
        return f"matvec_vectorized_{self.element_type}_{self.element_type}"

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[i32, i32, i32]
            + [op.inputs[1].type]  # A
            + [op.inputs[0].type]  # b
            + [op.output.type],  # c
            outputs=[],
        )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        k = prod(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        assert op.output is not None
        m = prod(cast(MemRefType[AnyDenseElement], op.output.type).get_shape())
        assert op.output is not None
        return [
            m := ConstantOp.from_int_and_width(m, i32),
            k := ConstantOp.from_int_and_width(k, i32),
            row_offset := ConstantOp.from_int_and_width(0, i32),
            CallOp(self.function_name, [m, k, row_offset, op.inputs[1], op.inputs[0], op.output], []),
        ]
