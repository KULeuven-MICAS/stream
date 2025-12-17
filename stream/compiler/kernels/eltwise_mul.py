from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AnyDenseElement,
    FunctionType,
    IndexType,
    MemRefType,
    i32,
)
from xdsl.dialects.func import CallOp
from xdsl.dialects.memref import ExtractAlignedPointerAsIndexOp
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import AIEKernel


@dataclass
class EltwiseMulKernel(AIEKernel):
    element_type: AnyDenseElement

    @property
    def linkwith_name(self) -> str:
        return "mul.o"

    @property
    def function_name(self) -> str:
        return f"eltwise_mul_{self.element_type}_vector"

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.outputs is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type, op.inputs[1].type, op.outputs.type, i32],
            outputs=[],
        )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        len = prod(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        assert op.outputs is not None
        return [
            len := ConstantOp.from_int_and_width(len, i32),
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], op.outputs, len], []),
        ]
