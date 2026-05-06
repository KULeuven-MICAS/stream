from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
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
from stream.compiler.kernels.aie_kernel import AIEKernel


@dataclass
class EltwiseMulKernel(AIEKernel):
    element_type: AnyDenseElement

    @property
    def linkwith_name(self) -> str:
        return "mul.o"

    @property
    def function_name(self) -> str:
        return f"eltwise_mul_{self.element_type}_scalar"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        # Intrinsic dimensions:
        r = 4  # ~m
        s = 8  # ~k  # noqa: F841
        t = 8  # ~n
        # Tiled kernel dimensions:
        mt = 32 // r
        nt = 64 // t
        return [
            TiledStridedLayout(
                [
                    TiledStride([Stride(r * t * nt, mt), Stride(t, r)]),
                    TiledStride([Stride(r * t, nt), Stride(1, t)]),
                ]
            )
            for _ in range(3)
        ]

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
