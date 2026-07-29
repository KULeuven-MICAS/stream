from xdsl.dialects.builtin import bf16

from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.matvec import MatVecKernel
from stream.compiler.kernels.silu import SiluKernel

AIEKernels = {
    "matvec": lambda utilization: MatVecKernel(utilization, bf16),
    # m and n are the kernel tile a mapping may leave unset.
    "silu": lambda utilization, layout, m=32, n=64, bfp16_mmul=False: SiluKernel(
        utilization, bf16, m, n, layout, bfp16_mmul
    ),
    "eltwise_mul": lambda utilization, layout, m=32, n=64, bfp16_mmul=False: EltwiseMulKernel(
        utilization, bf16, m, n, layout, bfp16_mmul
    ),
    "gemm": lambda utilization, m, k, n, layout, bfp16_mmul=False: GemmKernel(
        utilization, bf16, m, k, n, layout, bfp16_mmul
    ),
}
