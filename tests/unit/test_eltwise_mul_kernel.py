"""The elementwise multiply vectorizes when its tile leaves no remainder.

The vectorized kernel steps a whole vector per iteration and has no epilogue, so a
tile that is not a multiple of the vector width would silently leave its tail
unwritten. Layout does not enter into it: the multiply is position independent and
all three operands share a layout.
"""

from __future__ import annotations

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")


def kernel(m: int, n: int, layout: str = "default"):
    from xdsl.dialects.builtin import bf16

    from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel

    return EltwiseMulKernel(1.0, bf16, m, n, layout)


@pytest.mark.parametrize("layout", ["default", "contiguous"])
def test_a_whole_tile_vectorizes(layout: str):
    assert kernel(32, 64, layout).function_name == "eltwise_mul_bf16_vector"


def test_a_tile_with_a_remainder_stays_scalar():
    assert kernel(1, 15).function_name == "eltwise_mul_bf16_scalar"


def test_both_variants_are_linked_from_the_same_object():
    assert kernel(32, 64).linkwith_name == kernel(1, 1).linkwith_name == "mul.o"
