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
    assert kernel(m, n).linkwith_name == "mul.o"


def test_a_call_links_the_object_built_for_the_tile_it_takes(tmp_path):
    """A call handed half its declared row links the half-row object: an object compiled for a
    fixed element count would otherwise run past the tile it was given."""
    from types import SimpleNamespace

    from xdsl.dialects.builtin import MemRefType, bf16

    from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
    from stream.compiler.kernels.library import KernelLibrary

    toml = (Path(__file__).parents[2] / "stream/inputs/aie/kernels/aie2p.toml").read_text()
    sized = tmp_path / "sized.toml"
    sized.write_text(toml.replace('object = "mul.o"', 'object = "mul_{m}x{n}.o"'))
    multiply = EltwiseMulKernel(m=1, n=4096, layout="contiguous", library=KernelLibrary.load(sized))
    tile = SimpleNamespace(type=MemRefType(bf16, [1, 2048]))
    assert multiply.linkwith_name == "mul_1x4096.o"
    assert multiply.call_object(SimpleNamespace(inputs=[tile, tile, tile])) == "mul_1x2048.o"
