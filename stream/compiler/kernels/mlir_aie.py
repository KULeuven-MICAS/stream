"""The AIE kernels stream calls, bound to mlir-aie's ``aie.iron.kernels`` factories.

Imported only when codegen binds a call; each provider takes the target ``npu`` and the call's dimensions.
"""

from typing import get_args

import numpy as np
from aie.iron import ExternalFunction
from aie.iron.device import AIEArch, from_name
from aie.iron.kernels import activation, datamovement, eltwise, linalg
from aie.utils import set_current_device
from ml_dtypes import bfloat16

from stream.compiler.kernels.flash import SCALE_ROWS

INDEX = np.ndarray[(3,), np.dtype[np.int32]]


def _tile(size: int, dtype: type = bfloat16) -> type[np.ndarray]:
    return np.ndarray[(size,), np.dtype[dtype]]


def _scale(rows: int) -> type[np.ndarray]:
    return _tile(SCALE_ROWS * rows, np.float32)


def _target(npu: str) -> bool:
    """Bind ``npu`` as the device the factories build for; whether it is AIE2P."""
    device = from_name(npu, n_cols=None)
    set_current_device(device)
    return device.arch is AIEArch.AIE2p


def mm(npu: str, m: int, k: int, n: int, bfp16: bool = True, accumulate: bool = False) -> linalg.MatrixKernel:
    """mm.cc as IRON's GEMM builds it, its multiplies in bfp16 when ``bfp16`` (``Precision.MEDIUM``), else bf16; where
    it ``accumulate``s, into a float tile, with the cast of that tile to the bf16 output beside it."""
    _target(npu)
    output = np.float32 if accumulate else bfloat16
    kernel = linalg.mm(m, k, n, bfloat16, output, emulate_bf16_mmul_with_bfp16=bfp16, round_conv_even=True)
    if accumulate:
        kernel.cast = datamovement.convert_copy(m * n)
    return kernel


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


def _mha(npu: str, m: int, head: int, keys: int, causal: bool, **kind) -> linalg.MatrixKernel:
    """mha.cc at an m-query block of ``keys`` keys and a ``head``-wide head, its mask causal or none, built with the
    flags IRON's attention builds it with: bf16 multiplies, K as stored, and on AIE2P the accurate exp2 in place of
    the instruction, which is up to 6% high."""
    aie2p = _target(npu)
    return linalg.mha(dim_m=m, dim_k=head, dim_n=keys, b_col_maj=True, accurate_exp2=aie2p, causal=causal, **kind)


def partial_softmax(npu: str, m: int, n: int, causal: bool = True) -> Entry:
    """An online-softmax step over an m-query by n-key block; it reads no head, so its object takes the key block's."""
    scores = _tile(m * n)
    return Entry(
        _mha(npu, m, n, n, causal), "partial_softmax", [scores, scores, _scale(m), INDEX, bfloat16, *[np.int32] * 4], m
    )


def matmul_pv(npu: str, m: int, k: int, n: int, causal: bool = True) -> Entry:
    """``O += P V`` into a float accumulator, as mlir-aie declares it: P m queries by k keys, V k keys by n."""
    pv = _mha(npu, m, n, k, causal, pv=True, float_o=True)
    return Entry(pv, "matmul_PV", pv.arg_types(), m)


def softmax_rows(npu: str, m: int, n: int) -> Entry:
    _target(npu)
    return Entry(activation.softmax(n), "softmax_rows_bf16", [_tile(m * n), _tile(m * n), np.int32, np.int32])


def softmax_rows_causal(npu: str, m: int, n: int) -> Entry:
    rows = softmax_rows(npu, m, n)
    return Entry(rows.owner, "softmax_rows_causal_bf16", [*rows.arg_types(), np.int32])
