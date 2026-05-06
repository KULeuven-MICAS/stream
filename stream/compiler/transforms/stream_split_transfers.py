from dataclasses import dataclass, field
from typing import cast

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

from stream.compiler.dialects.stream import ChannelOp, OutEdgeOp, PullOp, PushOp, TransferOp


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
        for i, input in enumerate(op.inputs):
            offsets = list(cast(tuple[int, ...], op.spatial_strides.get_values()))
            if len(offsets) == 0:
                offsets = [0]
            push = PushOp(
                input,
                channel,
                op.ssis.data,
                (offsets[i],),
                # offsets
                op.sizes,
                op.strides,
                op.memtile,
                op.operand_indeces,
            )
            ops_to_add.append(push)
        for i, result in enumerate(op.results):
            for use in list(result.uses):
                if isinstance(edge := use.operation, OutEdgeOp):
                    for input in edge.inputs:
                        assert isinstance(input, OpResult)
                        if isinstance(input.op, TransferOp) and input.op.memtile == op.memtile:
                            self.channel_map[input.op] = channel
                offsets = list(cast(tuple[int, ...], op.spatial_strides.get_values()))
                if len(offsets) == 0:
                    offsets = [0]
                ops_to_add.append(
                    pull := PullOp(
                        op.outputs[0].type,
                        channel,
                        op.ssis.data,
                        (offsets[i],),
                        op.sizes,
                        op.strides,
                        op.memtile,
                        op.operand_indeces,
                    )
                )
                use.operation.operands[use.index] = pull.results[0]
        rewriter.insert_op(ops_to_add, InsertPoint.before(op))
        rewriter.erase_matched_op()


class StreamSplitTransfersPass(ModulePass):
    name = "stream-split-transfers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SplitTransferPattern(), apply_recursively=False).rewrite_module(op)
