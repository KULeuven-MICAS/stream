from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from snaxc.dialects.snax import LayoutCast
from snaxc.ir.tsl import TiledStridedLayout
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import IndexSwitchOp, YieldOp
from xdsl.ir import Operation, OpResult, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import CoreOp, DeviceOp, ObjectFIFOSubviewAccessOp

from stream.compiler.dialects.stream import ComputationNodeOp


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

        SymbolTable.insert_or_update(device_op, FuncOp(self.function_name, self.function_type(op), Region(), "private"))

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
        assert op.output is not None
        return CallOp(self.zero_name, [op.output], [])

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        # find device op to insert zero call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, FuncOp(self.zero_name, self.zero_type(op), Region(), "private"))

        # Insert zeroing after definition of output
        assert isinstance(op.output, OpResult)
        first_def: Operation = op.output.op

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
