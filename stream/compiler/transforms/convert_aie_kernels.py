from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels import AIEKernels
from stream.compiler.kernels.aie_kernel import AIEKernel


@dataclass
class ConvertAIEKernels(RewritePattern):
    kernels: dict[str, AIEKernel]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        aie_kernel = AIEKernels.get(op.kernel.data)
        if aie_kernel is not None:
            aie_kernel.rewrite(op, rewriter)
