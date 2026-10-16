"""The elementwise multiply calls mlir-aie's runtime-sized entry point, which handles any tail."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")


def kernel(m: int, n: int, layout: str = "default"):
    from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
    from stream.compiler.kernels.library import KernelLibrary

    library = KernelLibrary.load(Path(__file__).parents[2] / "stream/inputs/aie/kernels/aie2p.toml")
    return EltwiseMulKernel(m=m, n=n, layout=layout, library=library)


@pytest.mark.parametrize("m, n", [(32, 64), (1, 15)])
def test_every_tile_calls_the_runtime_sized_multiply(m: int, n: int):
    assert kernel(m, n).function_name == "eltwise_mul_bf16_vector_size"


def test_a_call_is_bound_at_the_tile_it_takes():
    """A call handed half its declared row is bound for the half row: an object compiled for a
    fixed element count would otherwise run past the tile it was given."""
    from types import SimpleNamespace

    from xdsl.dialects.builtin import MemRefType, bf16

    tile = SimpleNamespace(type=MemRefType(bf16, [1, 2048]))
    multiply = kernel(1, 4096, "contiguous")
    assert multiply.call_shape() == {"n": 4096, "m": 1}
    assert multiply.call_dims(SimpleNamespace(inputs=[tile, tile, tile])) == {"n": 2048, "m": 1}
