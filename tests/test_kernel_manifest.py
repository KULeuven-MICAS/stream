"""The registry, and the two ways a kernel library declares what it compiles."""

import pytest

from stream.compiler.kernels import manifest
from stream.compiler.kernels.flash import FlashKernel, PartialSoftmaxKernel
from stream.compiler.kernels.gemm import GemmKernel

LIBRARY = {
    "matmul_bf16_bf16": {
        "divisor": {"m": 16, "k": 8, "n": 16},
        "cycles": {"64,64,64": 1595.0, "per_op": {"cycles": 1595.0, "ops": 262144}},
    },
    "partial_softmax": {"blocks": {"m": [32, 64]}, "fixed": {"n": 64},
                        "cycles": {"32,64": 2323.0}},
    "matmul_PV": {"blocks": {"m": [32, 64]}, "fixed": {"k": 64, "n": 64},
                  "cycles": {"32,64,64": 859.0}},
}

SHAPE = dict(element_type="bf16", bfp16_mmul=True, utilization=61.8, layout="default")


@pytest.fixture
def library():
    manifest.adopt(LIBRARY)
    yield
    manifest.adopt({})


def test_a_divisor_offers_the_size_and_its_halvings(library):
    assert manifest.blocks("matmul_bf16_bf16", dict(m=64, k=64, n=64)) == {
        0: (16, 32, 64), 1: (8, 16, 32, 64), 2: (16, 32, 64)
    }


def test_a_divisor_is_relative_to_what_the_design_asks_for(library):
    """A source taking any multiple of 16 still only offers what divides this call."""
    assert manifest.blocks("matmul_bf16_bf16", dict(m=32, k=64, n=64))[0] == (16, 32)


def test_an_explicit_list_is_taken_as_given(library):
    assert manifest.blocks("partial_softmax", dict(m=32, n=64)) == {0: (32, 64)}


def test_a_shape_nobody_measured_falls_back_to_the_per_operation_anchor(library):
    assert manifest.cycles("matmul_bf16_bf16", dict(m=64, k=64, n=64)) == 1595.0
    assert manifest.cycles("matmul_bf16_bf16", dict(m=16, k=16, n=16)) == (1595.0, 262144)


def test_an_unknown_symbol_offers_nothing(library):
    assert manifest.blocks("nothing", dict(m=64)) == {}
    assert manifest.cycles("nothing", dict(m=64)) is None


def test_no_library_means_no_blocks_rather_than_a_crash():
    manifest.adopt({})
    assert GemmKernel(m=64, k=64, n=64, **SHAPE).block_sizes() == {}


def test_each_mha_symbol_reads_its_own_entry_not_the_gemm_base(library):
    """FlashKernel derives from GemmKernel, so without an override it would be priced as
    mm.cc's matmul and silently take the wrong anchor."""
    flash = FlashKernel(m=32, k=64, n=64, **SHAPE)
    softmax = PartialSoftmaxKernel(m=32, n=64, **{**SHAPE, "layout": "contiguous"})
    assert flash.manifest_key == "matmul_PV"
    assert softmax.manifest_key == "partial_softmax"
    assert manifest.cycles(flash.manifest_key, flash.call_shape()) == 859.0


def test_a_shape_the_library_does_not_compile_is_refused(library):
    with pytest.raises(ValueError, match="compiles m of"):
        FlashKernel(m=48, k=64, n=64, **SHAPE)
    with pytest.raises(ValueError, match="compiled with n=64"):
        FlashKernel(m=32, k=64, n=128, **SHAPE)
