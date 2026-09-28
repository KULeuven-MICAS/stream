import pytest

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.compiler.kernels.registry import AIE_KERNELS  # noqa: E402

LIBRARY = KernelLibrary.from_dict(
    {
        "family": {"matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}}},
        "kernel": {
            "matmul_bf16_bf16": {
                "family": "matmul",
                "object": "mm_{m}_{k}_{n}.o",
                "dims": [{"name": "k"}, {"name": "n"}, {"name": "m", "blocks": [16, 32]}],
            }
        },
    }
)


def test_a_kernel_binds_the_library_dimensions_to_its_node_dimensions():
    gemm = AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default", library=LIBRARY)
    assert [(position, size, d.name) for position, size, d in gemm.call_tile()] == [
        (1, 64, "k"),
        (2, 16, "n"),
        (0, 32, "m"),
    ]
    assert gemm.spec.symbol == "matmul_bf16_bf16"


def test_the_object_a_kernel_links_is_named_by_the_library():
    assert AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default", library=LIBRARY).linkwith_name == "mm_32_64_16.o"


def test_a_shape_the_library_does_not_compile_is_rejected():
    with pytest.raises(ValueError, match="m=64"):
        AIE_KERNELS["gemm"](m=64, k=64, n=16, layout="default", library=LIBRARY).validate()


def test_a_kernel_without_a_library_says_so():
    with pytest.raises(ValueError, match="needs a kernel library"):
        AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default").validate()
