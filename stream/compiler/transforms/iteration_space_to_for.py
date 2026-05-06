from collections.abc import Sequence

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.csl import RewritePattern
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block
from xdsl.irdl import OpResult
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl_aie.dialects.aie import CoreOp, EndOp, TileOp

from stream.compiler.dialects.stream import (
    ComputationNodeOp,
    PullOp,
    PushOp,
    StrensorType,
    StrensorVarAttr,
    StrensorVarType,
)
from stream.datatypes import LayerDim


def iteration_space_to_for(block: Block, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
    ops: Sequence[PushOp | PullOp | ComputationNodeOp] = []
    ssis = None

    relevant_dims: set[LayerDim] = set()
    for op in block.ops:
        if isinstance(op, ComputationNodeOp):
            if ssis is not None:  # if already set
                raise RuntimeError("only supporting one compute node per core")
            assert isinstance(output_type := op.output.type, StrensorType)
            ssis = output_type.ssis.data
            for var in ssis.get_kernel_variables():
                relevant_dims.add(var.dim)
            for input in op.inputs:
                assert isinstance(input_type := input.type, StrensorType)
                for var in input_type.ssis.data.get_kernel_variables():
                    relevant_dims.add(var.dim)
        if isinstance(op, PushOp | PullOp | ComputationNodeOp):
            ops.append(op)
        elif isinstance(op, EndOp):
            pass
        else:
            raise RuntimeError("non-steady state op encountered")
    assert ssis is not None

    # some common for loop variables
    lb = ConstantOp.from_int_and_width(0, IndexType())
    step = ConstantOp.from_int_and_width(1, IndexType())
    rewriter.insert_op((lb, step), InsertPoint.at_start(block))

    # create for loop nest:
    for_ops = []
    innermost = None
    for var in reversed(ssis.vars):
        if var.type != StrensorVarType.TEMPORAL:
            continue
        if var.dim not in relevant_dims:
            continue
        ub = ConstantOp.from_int_and_width(var.size, IndexType())
        for_op = ForOp(lb, ub, step, [], Block([*for_ops, YieldOp()], arg_types=[IndexType()]))
        if innermost is None:
            innermost = for_op
        for_op.attributes["layer_dim"] = StrensorVarAttr(var)
        for_ops = [ub, for_op]
    assert innermost is not None

    # insert for loop nest
    rewriter.insert_op(for_ops, InsertPoint.after(step))

    # insert ops into innermost for loop:
    for op in ops:
        op.detach()
    rewriter.insert_op(ops, InsertPoint.at_start(innermost.body.block))


class ComputeCoreToFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        assert isinstance(op.tile, OpResult) and isinstance(op.tile.op, TileOp)
        if op.tile.op.row.value.data > 1:
            iteration_space_to_for(op.region.block, rewriter)


class IterationSpaceToFor(ModulePass):
    """
    Converts iteration spaces to for loops
    """

    name = "iteration-space-to-for"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ComputeCoreToFor(), apply_recursively=False).rewrite_module(op)
