from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.dialects.func import FuncOp
from xdsl.ir import Operation, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import CoreOp, DeviceOp

from stream.compiler.dialects.stream import ComputationNodeOp


@dataclass
class AIEKernel(ABC):
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
