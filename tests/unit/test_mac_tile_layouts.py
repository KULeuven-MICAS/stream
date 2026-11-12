"""Operand layouts follow the MAC tile the kernel library declares for the matmul unit."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from stream.compiler.kernels.gemm import GemmKernel  # noqa: E402
from stream.compiler.kernels.library import Family, KernelLibrary  # noqa: E402
from stream.compiler.kernels.silu import SiluKernel  # noqa: E402

EXAMPLE = KernelLibrary.load(Path(__file__).parents[2] / "stream/inputs/aie/kernels/aie2p.toml")


def library(rows):
    families = {**EXAMPLE.families, "matmul": Family(151.0, {"m": rows, "k": 8, "n": 8})}
    return KernelLibrary(kernels=EXAMPLE.kernels, families=families)


def rows_of(layout):
    return layout.tstrides[0].strides[-1].bound


@pytest.mark.parametrize("rows", [4, 8])
def test_gemm_operands_follow_the_mac_tile(rows):
    a, _, c = GemmKernel(m=32, k=32, n=64, layout="default", library=library(rows)).operand_layouts()
    assert rows_of(a) == rows_of(c) == rows


@pytest.mark.parametrize("rows", [4, 8])
def test_an_elementwise_operand_keeps_the_tiling_a_gemm_leaves(rows):
    for layout in SiluKernel(layout="default", library=library(rows)).operand_layouts():
        assert rows_of(layout) == rows


def test_a_contiguous_elementwise_operand_is_not_tiled():
    for layout in SiluKernel(m=1, n=2048, layout="contiguous", library=EXAMPLE).operand_layouts():
        assert len(layout.tstrides[0].strides) == 1


def test_a_column_major_weight_is_read_in_n_by_k_mac_blocks():
    """mm.cc built with B_COL_MAJ reads B stored (n, k) as row-major MAC blocks of n by k."""
    from stream.compiler.kernels.aie_kernel import tiled_layout  # noqa: PLC0415

    kernel = GemmKernel(m=32, k=64, n=128, layout="default", b_col_maj=True, library=library(8))
    _, b, _ = kernel.operand_layouts()
    assert b == tiled_layout(128, 64, 8, 8)
    assert kernel.binding_options()["b_col_maj"] is True
    assert "b_col_maj" not in GemmKernel(m=32, k=64, n=128, layout="default", library=EXAMPLE).binding_options()
