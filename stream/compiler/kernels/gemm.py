from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.builtin import (
    AnyDenseElement,
    FunctionType,
    MemRefType,
)
from xdsl.dialects.func import CallOp
from xdsl.irdl import Operation

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import MAC_ROWS_BFP16, AIEKernelWithZeroing


@dataclass
class GemmKernel(AIEKernelWithZeroing):
    element_type: AnyDenseElement
    m: int
    k: int
    n: int
    layout: str
    bfp16_mmul: bool = False

    @property
    def zero_name(self) -> str:
        return f"zero_{self.element_type}_{self.m}_{self.k}_{self.n}"

    def zero_type(self, op: ComputationNodeOp) -> FunctionType:
        return FunctionType.from_lists(inputs=[op.inputs[2].type], outputs=[])

    @property
    def linkwith_name(self) -> str:
        return f"mm_{self.m}_{self.k}_{self.n}.o"

    @property
    def function_name(self) -> str:
        return f"matmul_{self.element_type}_{self.element_type}_{self.m}_{self.k}_{self.n}"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        # Intrinsic dimensions of the MAC the kernel was built for. mm.cc takes
        # 8x8x8 when bf16 matmuls are emulated on the bfp16 MACs and 4x8x8 when
        # they are not, so this has to agree with how the object was compiled.
        r = MAC_ROWS_BFP16 if self.bfp16_mmul else 4  # ~m
        s = 8  # ~k
        t = 8  # ~n
        # Tiled kernel dimensions:
        mt = self.m // r
        kt = self.k // s
        nt = self.n // t
        return [
            # A: mxk, tiles of rxs
            TiledStridedLayout(
                [
                    TiledStride([Stride(r * s * kt, mt), Stride(s, r)]),
                    TiledStride([Stride(r * s, kt), Stride(1, s)]),
                ]
            ),
            # B: kxn, tiles of sxt
            TiledStridedLayout(
                [
                    TiledStride([Stride(s * t * nt, kt), Stride(t, s)]),
                    TiledStride([Stride(s * t, nt), Stride(1, t)]),
                ]
            ),
            # C: mxn, tiles of rxt
            TiledStridedLayout(
                [
                    TiledStride([Stride(r * t * nt, mt), Stride(t, r)]),
                    TiledStride([Stride(r * t, nt), Stride(1, t)]),
                ]
            ),
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
