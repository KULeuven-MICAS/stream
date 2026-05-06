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
            len(op.inputs) > 1 and len(op.results) > 1
            # and len(op.inputs) == len(op.results) (internal join structure has this)
            # and op.ssis == op.ssis_dest
            # #TODO: fix this condition for more complicated stuff in the future
            # that should also cover the ssis dest of an edge op being none
        ):
            return
        if len(op.inputs) != len(op.outputs):
            # other case is not supported yet
            assert len(op.outputs) < len(op.inputs)
            # FIXME:
            # st_results = prod(
            #     var.size
            #     for var in op.ssis.data.variables
            #     if var.type is IterationVariableType.SPATIOTEMPORAL and var.relevant
            # )
            # assert len(op.outputs) * st_results == len(op.inputs)
            st_results = len(op.inputs) // len(op.outputs)
        else:
            st_results = 1

        for result in op.results:
            for use in result.uses:
                if isinstance(use.operation, OutEdgeOp):
                    return

        new_ops = []
        input_idx = 0
        for result in op.result_types:
            inputs = []
            for i in range(st_results):
                inputs.append(op.inputs[input_idx + i * len(op.outputs)])
            new_ops.append(
                TransferOp(
                    inputs,
                    (result,),
                    op.ssis.data,
                    op.offsets,
                    op.sizes,
                    op.strides,
                    op.spatial_strides,
                    op.memtile,
                    op.operand_indeces,
                )
            )
            input_idx += 1

        rewriter.replace_matched_op(new_ops, tuple(o.results[0] for o in new_ops))


class StreamSplitUnicastsPass(ModulePass):
    name = "stream-split-unicasts"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SplitUnicastsPattern()).rewrite_module(op)
