from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import ClassVar, cast

from snaxc.ir.tsl import TiledStridedLayout
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AnyDenseElement,
    MemRefType,
    i32,
)
from xdsl.dialects.func import CallOp
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import (
    AIEKernel,
    elementwise_operand_layout,
)


@dataclass(kw_only=True)
class EltwiseMulKernel(AIEKernel):
    OPERAND_AXES: ClassVar[Mapping[str, tuple[int, int]]] = {"m": (-1, -2), "n": (-1, -1)}
    m: int = 32
    n: int = 64
    layout: str

    @property
    def function_name(self) -> str:
        return f"eltwise_mul_{self.element_type}_vector_size"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return [elementwise_operand_layout(self.m, self.n, self.layout, self.mac) for _ in range(3)]

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        len = prod(cast(MemRefType[AnyDenseElement], op.inputs[0].type).get_shape())
        return [
            len := ConstantOp.from_int_and_width(len, i32),
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], op.inputs[2], len], []),
        ]
