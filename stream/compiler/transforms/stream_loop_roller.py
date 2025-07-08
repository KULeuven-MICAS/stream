from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from stream.compiler.dialects.stream import ComputationNodeOp, TransferOp


@dataclass
class LoopRollerPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.repeat.value.data == 1:
            return

        transfers_to_take_with: list[TransferOp] = []

        for operand in op.operands:
            # all operands are the result of transfers
            assert isinstance(operand, OpResult)
            assert isinstance(transfer_op := operand.op, TransferOp)

            if transfer_op.repeat.value.data == 1:
                # do not wrap in for loop
                continue

            # otherwise, repeat should be equal
            assert op.repeat == transfer_op.repeat
            transfers_to_take_with.append(transfer_op)

        # create for loop
        lb = ConstantOp.from_int_and_width(0, IndexType())
        ub = ConstantOp.from_int_and_width(op.repeat.value.data, IndexType())
        step = ConstantOp.from_int_and_width(1, IndexType())
        for_op = ForOp(lb, ub, step, [], Block([yield_op := YieldOp()], arg_types=[IndexType()]))

        rewriter.insert_op_after_matched_op([lb, ub, step, for_op])

        # move ops into for loop
        for op_to_move in transfers_to_take_with + [op]:
            op_to_move.detach()
            rewriter.insert_op(op_to_move, InsertPoint.before(yield_op))


class StreamLoopRollerPass(ModulePass):
    name = "stream-loop-roller"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LoopRollerPattern(), apply_recursively=False).rewrite_module(op)
