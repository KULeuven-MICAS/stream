from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from stream.compiler.dialects.stream import ChannelOp, EdgeOp, PullOp, PushOp, TransferOp


@dataclass
class SplitTransferPattern(RewritePattern):
    channel_map: dict[TransferOp, ChannelOp] = field(default_factory=dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter) -> None:
        ops_to_add: list[Operation] = []
        if op in self.channel_map:
            channel = self.channel_map[op]
        else:
            channel = ChannelOp()
            ops_to_add.append(channel)
        push = PushOp(
            op.input, channel, op.ssis.data, op.offsets, op.sizes, op.strides, op.spatial_strides, op.loop_dimensions
        )
        ops_to_add.append(push)
        for result in op.results:
            for i, use in enumerate(list(result.uses)):
                if isinstance(edge := use.operation, EdgeOp):
                    for input in edge.inputs:
                        assert isinstance(input, OpResult)
                        if isinstance(input.op, TransferOp):
                            self.channel_map[input.op] = channel
                ops_to_add.append(
                    pull := PullOp(
                        op.output[0].type,
                        channel,
                        op.ssis.data,
                        op.offsets,
                        op.sizes,
                        op.strides,
                        [int(x * i) for x in op.spatial_strides.iter_values()],
                        op.loop_dimensions,
                    )
                )
                use.operation.operands[use.index] = pull.results[0]
        rewriter.insert_op(ops_to_add, InsertPoint.before(op))
        rewriter.erase_matched_op()


class StreamSplitTransfersPass(ModulePass):
    name = "stream-split-transfers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SplitTransferPattern(), apply_recursively=False).rewrite_module(op)
