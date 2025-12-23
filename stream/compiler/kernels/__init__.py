from xdsl.dialects.builtin import bf16

from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.matvec import MatVecKernel
from stream.compiler.kernels.silu import SiluKernel

AIEKernels = {
    "matvec": lambda utilization: MatVecKernel(utilization, bf16),
    "silu": lambda utilization: SiluKernel(utilization, bf16),
    "eltwise_mul": lambda utilization: EltwiseMulKernel(utilization, bf16),
    "gemm": lambda utilization, m, k, n: GemmKernel(utilization, bf16, m, k, n),
}
