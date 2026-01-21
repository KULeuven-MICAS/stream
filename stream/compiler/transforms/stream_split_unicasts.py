from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from stream.compiler.dialects.stream import ChannelOp, OutEdgeOp, TransferOp


@dataclass
class SplitUnicastsPattern(RewritePattern):
    channel_map: dict[TransferOp, ChannelOp] = field(default_factory=dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter) -> None:
        if not (
            len(op.inputs) > 1 and len(op.results) > 1 and len(op.inputs) == len(op.results)
            # and op.ssis == op.ssis_dest
            # #TODO: fix this condition for more complicated stuff in the future
            # that should also cover the ssis dest of an edge op being none
        ):
            return
        for result in op.results:
            for use in result.uses:
                if isinstance(use.operation, OutEdgeOp):
                    return
        new_ops = [
            TransferOp(
                (input,),
                (result_type,),
                op.ssis.data,
                op.ssis_dest.data,
                op.offsets,
                op.sizes,
                op.strides,
                op.spatial_strides,
                op.memtile,
            )
            for (input, result_type) in zip(op.inputs, op.result_types, strict=True)
        ]
        rewriter.replace_matched_op(new_ops, tuple(o.results[0] for o in new_ops))


class StreamSplitUnicastsPass(ModulePass):
    name = "stream-split-unicasts"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SplitUnicastsPattern()).rewrite_module(op)
