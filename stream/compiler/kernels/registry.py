from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.flash import CausalGemmKernel, FlashKernel, FusedScoreSoftmaxKernel, PartialSoftmaxKernel
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.silu import SiluKernel
from stream.compiler.kernels.softmax import SoftmaxKernel


def gemm(*, flash: bool = False, causal: bool = False, **fields) -> GemmKernel:
    return (FlashKernel if flash else CausalGemmKernel if causal else GemmKernel)(**fields)


AIE_KERNELS = {
    "silu": SiluKernel,
    "eltwise_mul": EltwiseMulKernel,
    "softmax": SoftmaxKernel,
    "gemm": gemm,
    "partial_softmax": PartialSoftmaxKernel,
    "matmul_softmax": FusedScoreSoftmaxKernel,
}
