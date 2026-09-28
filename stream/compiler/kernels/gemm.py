from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

from snaxc.ir.tsl import TiledStridedLayout
from xdsl.dialects.builtin import (
    AnyDenseElement,
    FunctionType,
    MemRefType,
)
from xdsl.dialects.func import CallOp
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import AIEKernelWithZeroing, tiled_layout


@dataclass(kw_only=True)
class GemmKernel(AIEKernelWithZeroing):
    DIMS: ClassVar[Mapping[str, int]] = {"m": 0, "k": 1, "n": 2}
    m: int
    k: int
    n: int
    layout: str

    @property
    def zero_name(self) -> str:
        return f"zero_{self.element_type}_{self.m}_{self.k}_{self.n}"

    def zero_type(self, op: ComputationNodeOp) -> FunctionType:
        return FunctionType.from_lists(inputs=[op.inputs[2].type], outputs=[])

    @property
    def function_name(self) -> str:
        return f"matmul_{self.element_type}_{self.element_type}_{self.m}_{self.k}_{self.n}"

    @property
    def library_key(self) -> str:
        return f"matmul_{self.element_type}_{self.element_type}"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        r, s, t = self.mac["m"], self.mac["k"], self.mac["n"]
        return [
            tiled_layout(self.m, self.k, r, s),
            tiled_layout(self.k, self.n, s, t),
            tiled_layout(self.m, self.n, r, t),
        ]

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        assert op.output is not None
        return FunctionType.from_lists(
            inputs=[op.inputs[0].type]  # A
            + [op.inputs[1].type]  # b
            + [op.inputs[2].type],  # c
            outputs=[],
        )

    def check_operands(self, op: ComputationNodeOp) -> None:
        """The object file is compiled for one tile, so a mapping that tiles the loop
        differently would read past its operands."""
        assert op.output is not None
        for operand, expected in zip(op.inputs, ((self.m, self.k), (self.k, self.n), (self.m, self.n)), strict=True):
            shape = tuple(cast(MemRefType[AnyDenseElement], operand.type).get_shape())
            if shape != expected:
                raise ValueError(
                    f"kernel {self.function_name} takes a {expected[0]}x{expected[1]} operand "
                    f"but the tiling gives it {shape}"
                )

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        self.check_operands(op)
        return [
            CallOp(self.function_name, [op.inputs[0], op.inputs[1], op.inputs[2]], []),
        ]
