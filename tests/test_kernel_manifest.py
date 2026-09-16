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


def test_a_divisor_offers_nothing_because_a_rule_is_not_a_menu(library):
    """mm.cc takes any multiple of its divisor; that is legality, not a list of builds.

    Read as an offer it let the block search invent k=8 for the fused SwiGLU -- a block
    nobody compiled or timed. The per-operation anchor prices it at the same cycles as
    k=64 while making every transfer eight times smaller, so the search preferred it, and
    the 512-deep reduction became 64 rounds of accumulation into a bf16 buffer: 4.6x
    slower on device and 77% of elements outside tolerance against 14.5% before.
    """
    assert manifest.blocks("matmul_bf16_bf16") == {}
    assert manifest.divisors("matmul_bf16_bf16") == {0: 16, 1: 8, 2: 16}


def test_an_explicit_list_is_taken_as_given(library):
    assert manifest.blocks("partial_softmax") == {0: (32, 64)}
    assert manifest.divisors("partial_softmax") == {}


def test_a_shape_nobody_measured_falls_back_to_the_per_operation_anchor(library):
    assert manifest.cycles("matmul_bf16_bf16", dict(m=64, k=64, n=64)) == 1595.0
    assert manifest.cycles("matmul_bf16_bf16", dict(m=16, k=16, n=16)) == (1595.0, 262144)


def test_an_unknown_symbol_offers_nothing(library):
    assert manifest.blocks("nothing") == {}
    assert manifest.divisors("nothing") == {}
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


def test_a_gemm_still_declares_the_shape_it_is_compiled_at():
    """The granule is the kernel's own m, k and n, not a fact about any library.

    When the measured sizes and costs moved into the manifest, this went with them by
    mistake, and GemmKernel was left declaring nothing. Nothing crashed: a group that
    declares no intra-core tiling is tiled at its kernels' granules, so the GEMM's
    dimensions simply never entered the tiling and stayed at full extent, which put every
    core in the array three to six times over capacity. The regression is silent at the
    kernel and only shows up as an infeasible allocation, so it is pinned here.
    """
    manifest.adopt({})
    assert dict(GemmKernel(m=64, k=128, n=256, **SHAPE).granule()) == {0: 64, 1: 128, 2: 256}


def test_every_kernel_puts_a_floor_under_each_dimension_it_is_compiled_at():
    """A kernel that names a call dimension has to put a granule under it.

    Any dimension a kernel compiles into its block but leaves out of its granule is a
    dimension a fused group will tile at full extent. This is the invariant the GEMM
    broke; it is stated over every kernel so the next one cannot break it quietly.

    The comparison is by size rather than by position: a granule position indexes the
    node's dimensions, which coincide with the manifest's m, k, n for a GEMM but not for
    a two-dimensional kernel, where position 1 is n.
    """
    manifest.adopt({})
    contiguous = {**SHAPE, "layout": "contiguous"}
    for kernel in (
        GemmKernel(m=64, k=128, n=256, **SHAPE),
        FlashKernel(m=32, k=64, n=64, **SHAPE),
        PartialSoftmaxKernel(m=32, n=64, **contiguous),
    ):
        floors, shape = dict(kernel.granule()), kernel.call_shape()
        assert sorted(floors.values()) == sorted(shape.values()), (
            f"{type(kernel).__name__} is compiled at {shape} but its granule is {floors}, "
            f"so a fused group would tile the missing dimensions at full extent"
        )
