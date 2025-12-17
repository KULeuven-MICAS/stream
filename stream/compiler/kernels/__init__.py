from xdsl.dialects.builtin import bf16

from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.matvec import MatVecKernel
from stream.compiler.kernels.silu import SiluKernel

AIEKernels = {
    x.function_name: x
    for x in [
        MatVecKernel(bf16),
        SiluKernel(bf16),
        EltwiseMulKernel(bf16)
    ]
}
