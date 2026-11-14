from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.flash import CausalGemmKernel, FlashKernel, FusedScoreSoftmaxKernel, PartialSoftmaxKernel
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.silu import SiluKernel
from stream.compiler.kernels.softmax import SoftmaxKernel


def gemm(*, flash: bool = False, **fields) -> GemmKernel:
    """The value side of a flash step when ``flash``, causal unless told otherwise; else a GEMM, skipping the blocks
    past the diagonal when ``causal``."""
    if flash:
        return FlashKernel(**fields)
    return (CausalGemmKernel if fields.pop("causal", False) else GemmKernel)(**fields)


AIE_KERNELS = {
    "silu": SiluKernel,
    "eltwise_mul": EltwiseMulKernel,
    "softmax": SoftmaxKernel,
    "gemm": gemm,
    "partial_softmax": PartialSoftmaxKernel,
    "matmul_softmax": FusedScoreSoftmaxKernel,
}
POST_OP_KERNELS = ("silu", "eltwise_mul")
"""The elementwise kernels a GEMM runs as post-ops on its output tile."""
