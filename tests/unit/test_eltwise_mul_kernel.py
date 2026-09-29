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
