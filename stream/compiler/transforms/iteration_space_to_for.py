from collections.abc import Iterable, Sequence

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.csl import RewritePattern
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block, Operation
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
    ChannelOp,
    ComputationNodeOp,
    FusionGroupOp,
    PullOp,
    PushOp,
    StrensorType,
    StrensorVar,
    StrensorVarAttr,
    StrensorVarType,
    YieldOp as StreamYieldOp,
)
from stream.datatypes import LayerDim

# Ops that belong inside the steady-state loop body; everything else is hoisted or left.
LOOP_BODY_OPS = (PushOp, PullOp, ComputationNodeOp)
# Ops that may appear beside them without being part of an iteration.
LOOP_INVARIANT_OPS = (ChannelOp, EndOp, StreamYieldOp, ConstantOp)


def _kernel_dims(node: ComputationNodeOp) -> Iterable[LayerDim]:
    for value in (node.output, *node.inputs):
        if isinstance(value.type, StrensorType):
            for var in value.type.ssis.data.get_kernel_variables():
                yield var.dim


def _temporal_vars(nodes: Sequence[ComputationNodeOp]) -> Sequence[StrensorVar]:
    """Temporal variables the nodes iterate over, outermost first."""
    spaces = [n.output.type.ssis.data for n in nodes if isinstance(n.output.type, StrensorType)]
    if not spaces:
        return ()
    relevant = {dim for node in nodes for dim in _kernel_dims(node)}
    longest = max(spaces, key=lambda s: sum(v.type == StrensorVarType.TEMPORAL for v in s.vars))
    return [v for v in longest.vars
            if v.type == StrensorVarType.TEMPORAL and v.dim in relevant and v.size > 1]


def iteration_space_to_for(block: Block, rewriter: PatternRewriter):
    body: list[Operation] = []
    nodes: list[ComputationNodeOp] = []

    for op in block.ops:
        if isinstance(op, ComputationNodeOp):
            nodes.append(op)
        if isinstance(op, LOOP_BODY_OPS):
            body.append(op)
        elif not isinstance(op, LOOP_INVARIANT_OPS):
            raise RuntimeError(f"non-steady-state op in iteration space: {op.name}")

    temporal = _temporal_vars(nodes)
    if not temporal:
        return

    lb = ConstantOp.from_int_and_width(0, IndexType())
    step = ConstantOp.from_int_and_width(1, IndexType())
    rewriter.insert_op((lb, step), InsertPoint.at_start(block))

    for_ops: list[Operation] = []
    innermost = None
    for var in reversed(temporal):
        ub = ConstantOp.from_int_and_width(var.size, IndexType())
        for_op = ForOp(lb, ub, step, [], Block([*for_ops, YieldOp()], arg_types=[IndexType()]))
        for_op.attributes["layer_dim"] = StrensorVarAttr(var)
        innermost = innermost or for_op
        for_ops = [ub, for_op]

    assert innermost is not None
    rewriter.insert_op(for_ops, InsertPoint.after(step))
    for op in body:
        op.detach()
    rewriter.insert_op(body, InsertPoint.at_start(innermost.body.block))


class ComputeCoreToFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        assert isinstance(op.tile, OpResult) and isinstance(op.tile.op, TileOp)
        if op.tile.op.row.value.data > 1:
            iteration_space_to_for(op.region.block, rewriter)


class FusionGroupToFor(RewritePattern):
    """For backends that keep the group whole instead of splitting it across cores."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FusionGroupOp, rewriter: PatternRewriter):
        if any(isinstance(inner, ForOp) for inner in op.body.block.ops):
            return
        iteration_space_to_for(op.body.block, rewriter)


class IterationSpaceToFor(ModulePass):
    """Converts iteration spaces to for loops"""

    name = "iteration-space-to-for"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ComputeCoreToFor(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(FusionGroupToFor(), apply_recursively=False).rewrite_module(op)
