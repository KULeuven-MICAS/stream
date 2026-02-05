from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects.builtin import (
    AnyDenseElement,
    FunctionType,
)
from xdsl.dialects.func import CallOp
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import AIEKernel


@dataclass
class GemmKernel(AIEKernel):
    element_type: AnyDenseElement
    m: int
    k: int
    n: int

    @property
    def linkwith_name(self) -> str:
        return "mm.o"

    @property
    def function_name(self) -> str:
        return f"matmul_{self.element_type}_{self.element_type}"

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[1].type]  # A
            + [op.inputs[0].type]  # b
            + [op.output.type],  # c
            outputs=[],
        )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        assert op.output is not None
        return [
            CallOp(self.function_name, [op.inputs[1], op.inputs[0], op.output], []),
        ]
