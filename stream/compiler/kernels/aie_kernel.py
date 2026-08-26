from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from snaxc.dialects.snax import LayoutCast
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import CoreOp, DeviceOp, ObjectFIFOSubviewAccessOp

from stream.compiler.dialects.stream import ComputationNodeOp, StrensorVar, StrensorVarAttr

# Intrinsic MAC tile of the AIE2p kernels, and the layouts an operand can take.
# mm.cc takes 8 rows when bf16 matmuls run on the bfp16 MACs and 4 when they do not.
R, T = 4, 8
MAC_ROWS_BFP16 = 8
MAC_TILED = "default"
CONTIGUOUS = "contiguous"
VECTOR_LANES = 16
"""Elements a vectorized elementwise kernel loads per step."""


def elementwise_operand_layout(m: int, n: int, layout: str, mac_rows: int = R) -> TiledStridedLayout:
    """Layout of one operand of an elementwise kernel.

    An elementwise kernel walks its operands linearly, so it imposes no layout of
    its own; the layout only has to match what the operands already are.
    ``default`` is the r x t tiling a GEMM writes its output in, so an elementwise
    layer fused behind one needs no transformation. ``contiguous`` is plain row
    major, which is what a layer reading from and writing to memory wants: its
    transfers then run the length of a row instead of one MAC tile at a time.
    """
    if layout == CONTIGUOUS:
        return TiledStridedLayout([TiledStride([Stride(n, m)]), TiledStride([Stride(1, n)])])
    mt, nt = m // mac_rows, n // T
    return TiledStridedLayout(
        [
            TiledStride([Stride(mac_rows * T * nt, mt), Stride(T, mac_rows)]),
            TiledStride([Stride(mac_rows * T, nt), Stride(1, T)]),
        ]
    )


def induction_variable(op: Operation, var: StrensorVar, occurrence: int = 0) -> SSAValue:
    """The induction variable of the enclosing loop iterating ``var``.

    ``IterationSpaceToFor`` names every loop it creates after the variable it drives,
    which is the only way back from a kernel call to where in the iteration space it sits.
    A dimension split into parts of equal size gives loops that name themselves alike, so
    ``occurrence`` picks among them counting from the innermost out -- the order the parts
    themselves are read in.
    """
    parent, seen = op.parent_op(), 0
    while parent is not None:
        if isinstance(parent, ForOp) and parent.attributes.get("layer_dim") == StrensorVarAttr(var):
            if seen == occurrence:
                return parent.body.block.args[0]
            seen += 1
        parent = parent.parent_op()
    raise ValueError(f"kernel call is not inside the loop iterating {var} ({occurrence})")


@dataclass
class AIEKernel(ABC):
    utilization: float

    @property
    def unique_name(self) -> str:
        return self.function_name

    @property
    @abstractmethod
    def linkwith_name(self) -> str: ...

    @property
    @abstractmethod
    def function_name(self) -> str: ...

    @abstractmethod
    def function_type(self, op: ComputationNodeOp) -> FunctionType: ...

    @abstractmethod
    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]: ...

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return []

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        # find device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(
            device_op,
            FuncOp(self.function_name, self.function_type(op), Region(), "private"),
        )

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op.link_with = StringAttr(self.linkwith_name)

        # replace computation node with func call op
        rewriter.insert_op(self.function_call(op), InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class AIEKernelWithZeroing(AIEKernel, ABC):
    @property
    @abstractmethod
    def zero_name(self) -> str: ...

    @abstractmethod
    def zero_type(self, op: ComputationNodeOp) -> FunctionType: ...

    def zero_call(self, op: ObjectFIFOSubviewAccessOp) -> Operation:
        return CallOp(self.zero_name, [op.output], [])

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        # find device op to insert zero call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, FuncOp(self.zero_name, self.zero_type(op), Region(), "private"))

        # Insert zeroing after definition of output
        assert isinstance((last := op.inputs[-1]), OpResult)
        first_def: Operation = last.op

        # get rid of layout casts
        while isinstance(first_def, LayoutCast):
            assert isinstance(first_def.source, OpResult)
            first_def = first_def.source.op

        # handle index switch statements
        if isinstance(first_def, IndexSwitchOp):
            for case_region in first_def.case_regions:
                yield_op = case_region.block.last_op
                assert isinstance(yield_op, YieldOp)
                case_def = yield_op.arguments[0]
                assert isinstance(case_def, OpResult)
                case_def = case_def.op
                # again, get rid of layout casts:
                while isinstance(case_def, LayoutCast):
                    assert isinstance(case_def.source, OpResult)
                    case_def = case_def.source.op
                assert isinstance(case_def, ObjectFIFOSubviewAccessOp)
                rewriter.insert_op(self.zero_call(case_def), InsertPoint.after(case_def))
        else:
            assert isinstance(first_def, ObjectFIFOSubviewAccessOp)
            rewriter.insert_op(self.zero_call(first_def), InsertPoint.after(first_def))

        # Then, rewrite op as before:
        AIEKernel.rewrite(self, op, rewriter)
