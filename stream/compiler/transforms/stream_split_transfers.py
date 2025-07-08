from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from stream.compiler.dialects.stream import ChannelOp, PullOp, PushOp, TransferOp


@dataclass
class SplitTransferPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter) -> None:
        channel = ChannelOp()
        push = PushOp(op.input, channel, op.ssis.data, op.offsets, op.sizes, op.strides, op.loop_dimensions)
        pull = PullOp(op.output[0].type, channel, op.ssis.data, op.offsets, op.sizes, op.strides, op.loop_dimensions)
        op.output[0].replace_by(pull.output)
        rewriter.replace_matched_op([channel, push, pull])


class StreamSplitTransfersPass(ModulePass):
    name = "stream-split-transfers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SplitTransferPattern(), apply_recursively=False).rewrite_module(op)
