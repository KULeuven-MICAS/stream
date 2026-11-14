"""The AIE kernels stream calls, bound to mlir-aie's ``aie.iron.kernels`` factories.

Imported only when codegen binds a call; each provider takes the target ``npu`` and the call's dimensions.
"""

from typing import get_args

import numpy as np
from aie.iron import ExternalFunction
from aie.iron.device import from_name
from aie.iron.kernels import activation, eltwise, linalg
from aie.utils import set_current_device
from ml_dtypes import bfloat16

from stream.compiler.kernels.flash import SCALE_ROWS

INDEX = np.ndarray[(3,), np.dtype[np.int32]]


def _tile(size: int, dtype: type = bfloat16) -> type[np.ndarray]:
    return np.ndarray[(size,), np.dtype[dtype]]


def _scale(rows: int) -> type[np.ndarray]:
    return _tile(SCALE_ROWS * rows, np.float32)


def _target(npu: str) -> None:
    set_current_device(from_name(npu, n_cols=None))


def mm(npu: str, m: int, k: int, n: int) -> linalg.MatrixKernel:
    _target(npu)
    return linalg.mm(m, k, n, bfloat16, bfloat16, emulate_bf16_mmul_with_bfp16=True, round_conv_even=True)


def silu(npu: str, m: int, n: int) -> ExternalFunction:
    _target(npu)
    return activation.silu(m * n, use_lut=True)


def mul(npu: str, m: int, n: int) -> ExternalFunction:
    _target(npu)
    return eltwise.mul_sized(m * n)


class Entry(ExternalFunction):
    """An entry point of ``owner``'s object, with the calls the flash kernels make beside it on a
    ``rows``-query block, which mlir-aie builds but does not declare."""

    def __init__(self, owner: ExternalFunction, symbol: str, arg_types: list, rows: int = 0) -> None:
        super().__init__(
            symbol,
            object_file_name=owner.object_file_name,
            source_file=owner.source_file,
            arg_types=arg_types,
            include_dirs=owner.include_dirs,
            compile_flags=owner.compile_flags,
            symbol_prefix=owner.object_file.symbol_prefix,
        )
        self.owner, self.rows, self.scale = owner, rows, _scale(rows)

    @property
    def zero(self) -> ExternalFunction:
        return self.owner.zero

    @property
    def init_scale_buffer(self) -> "Entry":
        return Entry(self.owner, "init_scale_buffer", [self.scale, np.int32], self.rows)

    @property
    def rescale_O(self) -> "Entry":
        """The float accumulator divided by the row sums into a bf16 output of its size."""
        accumulator = self.owner.arg_types()[2]
        output = _tile(get_args(accumulator)[0][0])
        return Entry(self.owner, "rescale_O", [accumulator, output, self.scale, np.int32, INDEX], self.rows)

    @property
    def passThroughLine(self) -> "Entry":
        """The copy of 32-bit lanes that hands the running scale on, declared on the float buffers it copies."""
        copy = eltwise.passthrough(SCALE_ROWS * self.rows, np.int32)
        return Entry(copy, "passThroughLine", [self.scale, self.scale, np.int32], self.rows)


def _mha(npu: str, m: int, k: int, n: int) -> linalg.MatrixKernel:
    _target(npu)
    return linalg.mha(dim_m=m, dim_k=k, dim_n=n, emulate_bf16_mmul_with_bfp16=True)


def partial_softmax(npu: str, m: int, n: int) -> Entry:
    """An online-softmax step over an m-query by n-key block; mha.cc sizes the block by its head, so n is both."""
    scores = _tile(m * n)
    return Entry(
        _mha(npu, m, n, n), "partial_softmax", [scores, scores, _scale(m), INDEX, bfloat16, *[np.int32] * 4], m
    )


def matmul_pv(npu: str, m: int, k: int, n: int) -> Entry:
    """``O += P V`` into a float accumulator, as mlir-aie declares it: P m queries by k keys, V k keys by n."""
    _target(npu)
    pv = linalg.mha(dim_m=m, dim_k=n, dim_n=k, pv=True, float_o=True, emulate_bf16_mmul_with_bfp16=True)
    return Entry(pv, "matmul_PV", pv.arg_types(), m)


def softmax_rows(npu: str, m: int, n: int) -> Entry:
    _target(npu)
    return Entry(activation.softmax(n), "softmax_rows_bf16", [_tile(m * n), _tile(m * n), np.int32, np.int32])


def softmax_rows_causal(npu: str, m: int, n: int) -> Entry:
    rows = softmax_rows(npu, m, n)
    return Entry(rows.owner, "softmax_rows_causal_bf16", [*rows.arg_types(), np.int32])
