"""Operand layouts follow the MAC tile the kernel object was compiled for.

mm.cc takes an 8 row MAC tile when bf16 matmuls run on the bfp16 MACs and a 4 row one
when they do not. The layout the generated DMAs produce has to match, and an
elementwise layer that keeps the layout a neighbouring GEMM writes has to match it too.
"""

from __future__ import annotations

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")


def rows_of(layout):
    """The MAC tile rows a tiled layout addresses, from its innermost row stride."""
    return layout.tstrides[0].strides[-1].bound


def gemm(bfp16_mmul):
    from xdsl.dialects.builtin import bf16

    from stream.compiler.kernels.gemm import GemmKernel

    return GemmKernel(1.0, bf16, 32, 32, 64, "default", bfp16_mmul)


def elementwise(bfp16_mmul):
    from xdsl.dialects.builtin import bf16

    from stream.compiler.kernels.silu import SiluKernel

    return SiluKernel(1.0, bf16, 32, 64, "default", bfp16_mmul)


@pytest.mark.parametrize("bfp16_mmul, expected", [(False, 4), (True, 8)])
def test_gemm_operands_follow_the_mac_tile(bfp16_mmul, expected):
    a, _, c = gemm(bfp16_mmul).operand_layouts()
    assert rows_of(a) == expected
    assert rows_of(c) == expected


@pytest.mark.parametrize("bfp16_mmul, expected", [(False, 4), (True, 8)])
def test_elementwise_operands_follow_the_mac_tile(bfp16_mmul, expected):
    for layout in elementwise(bfp16_mmul).operand_layouts():
        assert rows_of(layout) == expected


def test_a_gemm_and_the_elementwise_beside_it_agree():
    _, _, c = gemm(True).operand_layouts()
    assert all(rows_of(x) == rows_of(c) for x in elementwise(True).operand_layouts())


def test_a_contiguous_elementwise_operand_is_not_tiled():
    from xdsl.dialects.builtin import bf16

    from stream.compiler.kernels.silu import SiluKernel

    for layout in SiluKernel(1.0, bf16, 1, 2048, "contiguous", True).operand_layouts():
        assert len(layout.tstrides[0].strides) == 1
