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

from stream.compiler.dialects.stream import ComputationNodeOp, TransferOp
from stream.workload.steady_state.iteration_space import IterationVariable, IterationVariableReuse


def disable_reuse(iv: IterationVariable) -> None:
    """
    Sets both MEM_TILE_NO_REUSE and COMPUTE_TILE_NO_REUSE on the given IterationVariable,
    while removing any previous reuse flags.
    """

    new_flags = IterationVariableReuse.MEM_TILE_NO_REUSE
    new_flags |= IterationVariableReuse.COMPUTE_TILE_NO_REUSE

    # Update the iteration variable
    iv._reuse = new_flags


@dataclass
class SetNoReusePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp | ComputationNodeOp, rewriter: PatternRewriter) -> None:
        ssis = op.ssis.data

        for iv in ssis.variables:
            disable_reuse(iv)


class SetNoReusePass(ModulePass):
    name = "set-no-reuse"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SetNoReusePattern(), apply_recursively=False).rewrite_module(op)
