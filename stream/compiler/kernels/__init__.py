from xdsl.dialects.builtin import bf16

from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.flash import (
    CausalGemmKernel,
    FlashKernel,
    FusedScoreSoftmaxKernel,
    PartialSoftmaxKernel,
)
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.matvec import MatVecKernel
from stream.compiler.kernels.silu import SiluKernel
from stream.compiler.kernels.softmax import SoftmaxKernel

AIEKernels = {
    "matvec": lambda utilization: MatVecKernel(utilization, bf16),
    # m and n are the kernel tile a mapping may leave unset.
    "silu": lambda utilization, layout, m=32, n=64, bfp16_mmul=False: SiluKernel(
        utilization, bf16, m, n, layout, bfp16_mmul
    ),
    "eltwise_mul": lambda utilization, layout, m=32, n=64, bfp16_mmul=False: EltwiseMulKernel(
        utilization, bf16, m, n, layout, bfp16_mmul
    ),
    # n is the row softmax reduces, which no default can guess.
    "softmax": lambda utilization, n, layout, m=1, bfp16_mmul=False, causal=False: SoftmaxKernel(
        utilization, bf16, m, n, layout, bfp16_mmul, causal
    ),
    # ``flash`` picks the gemm that carries an online softmax's running scale with it,
    # ``causal`` the one that skips the blocks a mask would zero.
    "gemm": lambda utilization, m, k, n, layout, bfp16_mmul=False, flash=False, causal=False: (
        FlashKernel if flash else CausalGemmKernel if causal else GemmKernel
    )(utilization, bf16, m, k, n, layout, bfp16_mmul),
    "partial_softmax": lambda utilization, n, layout, m=1, bfp16_mmul=False: PartialSoftmaxKernel(
        utilization, bf16, m, n, layout, bfp16_mmul
    ),
    # The same step with either side in the tiling a GEMM works in, for a softmax layer that
    # hands to or takes from the core beside it rather than through a memory tile.
    "partial_softmax_mode": lambda utilization, n, layout, m=1, bfp16_mmul=False, tiled_in=False, tiled_out=False: (
        PartialSoftmaxKernel(utilization, bf16, m, n, layout, bfp16_mmul, tiled_in=tiled_in, tiled_out=tiled_out)
    ),
    # The score GEMM and the online softmax on one core, which is what lets two fused
    # layers cover four rows where three layers leave one idle.
    "matmul_softmax": lambda utilization, m, k, n, layout, bfp16_mmul=False: FusedScoreSoftmaxKernel(
        utilization, bf16, m, k, n, layout, bfp16_mmul
    ),
}
