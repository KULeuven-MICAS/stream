"""The AIE kernels stream calls, bound to mlir-aie's ``aie.iron.kernels`` factories.

Imported only when codegen binds a call; each provider takes the target ``npu`` and the call's dimensions.
"""

import numpy as np
from aie.iron import ExternalFunction
from aie.iron.device import from_name
from aie.iron.kernels import activation, eltwise, linalg
from aie.utils import set_current_device
from ml_dtypes import bfloat16

INDEX = np.ndarray[(2,), np.dtype[np.int32]]


def _tile(size: int) -> type[np.ndarray]:
    return np.ndarray[(size,), np.dtype[bfloat16]]


def _target(npu: str) -> None:
    set_current_device(from_name(npu, n_cols=None))


def mm(npu: str, m: int, k: int, n: int) -> linalg.MatrixKernel:
    _target(npu)
    return linalg.mm(m, k, n, bfloat16, bfloat16, emulate_bf16_mmul_with_bfp16=True, round_conv_even=True)


def silu(npu: str, m: int, n: int) -> ExternalFunction:
    _target(npu)
    return activation.silu_sized(m * n)


def mul(npu: str, m: int, n: int) -> ExternalFunction:
    _target(npu)
    return eltwise.mul_sized(m * n)


class Entry(ExternalFunction):
    """An entry point of ``owner``'s object that mlir-aie builds but does not declare yet,
    with the calls the flash kernels make beside it on a ``rows``-query block."""

    def __init__(self, owner: ExternalFunction, rows: int, symbol: str, arg_types: list) -> None:
        super().__init__(
            symbol,
            object_file_name=owner.object_file_name,
            source_file=owner.source_file,
            arg_types=arg_types,
            include_dirs=owner.include_dirs,
            compile_flags=owner.compile_flags,
            symbol_prefix=owner.object_file.symbol_prefix,
        )
        self.owner, self.rows, self.scale = owner, rows, _tile(4 * rows)

    @property
    def zero(self) -> ExternalFunction:
        return self.owner.zero

    @property
    def init_scale_buffer(self) -> "Entry":
        return Entry(self.owner, self.rows, "init_scale_buffer", [self.scale, np.int32])

    @property
    def rescale_O(self) -> "Entry":
        return Entry(self.owner, self.rows, "rescale_O", [self.owner.arg_types()[2], self.scale, np.int32, INDEX])

    @property
    def passThroughLine(self) -> "Entry":
        """The copy of 16-bit lanes that hands the running scale on, declared on the bf16 buffers it copies."""
        copy = eltwise.passthrough(4 * self.rows, np.int16)
        return Entry(copy, self.rows, "passThroughLine", [self.scale, self.scale, np.int32])


def _mha(npu: str, m: int, k: int, n: int) -> linalg.MatrixKernel:
    _target(npu)
    return linalg.mha(dim_m=m, dim_k=k, dim_n=n, emulate_bf16_mmul_with_bfp16=True)


def partial_softmax(npu: str, m: int, n: int) -> Entry:
    """An online-softmax step over an m-query by n-key block; mha.cc sizes the block by its head, so n is both."""
    scores = _tile(m * n)
    return Entry(
        _mha(npu, m, n, n), m, "partial_softmax", [scores, scores, _tile(4 * m), INDEX, bfloat16, *[np.int32] * 4]
    )


def matmul_pv(npu: str, m: int, k: int, n: int) -> Entry:
    mha = _mha(npu, m, k, n)
    return Entry(mha, m, "matmul_PV", [*mha.arg_types()[:3], _tile(4 * m), np.int32, np.int32, INDEX, np.int32])
