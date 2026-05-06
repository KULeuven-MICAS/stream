from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from stream.compiler.dialects.stream import (
    ChannelOp,
    ComputationNodeOp,
    GatherOp,
    PullOp,
    PushOp,
    StrensorSpace,
    StrensorSpaceAttr,
    StrensorType,
    StrensorVar,
    StrensorVarType,
    TransferOp,
    YieldOp,
)


def iterate_spat_vars(
    spat_vars: Sequence[StrensorVar],
) -> Iterable[tuple[StrensorVar, ...]]:
    if len(spat_vars) == 0:
        yield tuple()
        return
    vars = ((StrensorVar(StrensorVarType.POINT, i, spat_vars[0].dim),) for i in range(spat_vars[0].size))
    for var in vars:
        if len(spat_vars) == 1:
            yield var
        else:
            for other_vars in iterate_spat_vars(spat_vars[1:]):
                yield var + other_vars


@dataclass
class UnrollComputationNodes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.spatial_index is not None:
            # already spatially unrolled
            return
        # computation nodes can be simply unrolled by their core allocations:
        new_ops = []
        output_tensor = op.output.type
        assert isinstance(output_tensor, StrensorType)
        spat_vars = list(output_tensor.ssis.data.get_spatial_variables())

        for spat_var, core in zip(
            iterate_spat_vars(spat_vars),
            output_tensor.core_allocation,
            strict=True,
        ):
            result = StrensorType(
                output_tensor.element_type,
                output_tensor.ssis,
                (core,),
                output_tensor.reuse_index,
            )
            new_ops.append(
                ComputationNodeOp(
                    op.inputs,
                    (result,),
                    op.kernel.data,
                    StrensorSpaceAttr(StrensorSpace(spat_var)),
                )
            )
        gather_op = GatherOp(new_ops, output_tensor)
        rewriter.replace_matched_op((*new_ops, gather_op))


@dataclass
class UnrollTransfers(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter) -> None:
        if op.parent_op() is None:
            breakpoint()
        ops: list[Operation] = []
        new_results: list[SSAValue] = []
        # create singular channel
        channel = ChannelOp()
        ops.append(channel)

        # inputs:
        assert len(op.inputs) == 1
        input = op.inputs[0]
        assert isinstance(input_type := input.type, StrensorType)
        input_spat_vars = list(input_type.ssis.data.get_spatial_variables())
        for var in iterate_spat_vars(input_spat_vars):
            ssis = StrensorSpaceAttr(StrensorSpace(tuple(var)))
            ops.append(PushOp(input, channel, ssis))

        # outputs:
        for output in op.outputs:
            assert isinstance(output.type, StrensorType)
            output_spat_vars = list(output.type.ssis.data.get_spatial_variables())
            pull_ops = []
            for core, var in zip(
                output.type.core_allocation,
                iterate_spat_vars(
                    output_spat_vars,
                ),
                strict=True,
            ):
                result_type = StrensorType(
                    output.type.element_type,
                    output.type.ssis,
                    (core,),
                    output.type.reuse_index,
                )
                ssis = StrensorSpaceAttr(StrensorSpace(tuple(var)))
                pull_ops.append(PullOp(result_type, channel, ssis))
            gather_op = GatherOp(pull_ops, output.type)
            ops.extend(pull_ops)
            ops.append(gather_op)
            new_results.append(gather_op.output)
        rewriter.replace_matched_op(ops, new_results)


@dataclass
class SquashGatherOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GatherOp, rewriter: PatternRewriter) -> None:
        if len(op.inputs) != len(op.output.uses):
            breakpoint()
        assert len(op.inputs) == len(op.output.uses)
        inputs: dict[StrensorSpace, SSAValue | Operation] = {}
        # gather mapping spatial index -> input
        for input in op.inputs:
            assert isinstance(input, OpResult) and isinstance(in_op := input.op, PullOp | ComputationNodeOp)
            assert in_op.spatial_index is not None
            inputs[in_op.spatial_index.data] = input
        # rewire outputs to use correct input
        for output in op.output.uses.copy():
            if isinstance(output.operation, YieldOp):
                assert len(inputs) == 1
                output.operation.operands[output.index] = SSAValue.get(next(iter(inputs.values())))
                continue
            assert isinstance(out_op := output.operation, PushOp | ComputationNodeOp)
            assert out_op.spatial_index is not None
            assert out_op.spatial_index.data in inputs
            out_op.operands[output.index] = SSAValue.get(inputs[out_op.spatial_index.data])
        rewriter.erase_matched_op()


class SpatialUnrollPass(ModulePass):
    """
    Resolve spatial unrolling of stream steady state
    """

    name = "unroll"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(UnrollComputationNodes()).rewrite_module(op)
        PatternRewriteWalker(UnrollTransfers()).rewrite_module(op)
        PatternRewriteWalker(SquashGatherOps()).rewrite_module(op)
