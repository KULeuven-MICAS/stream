import pytest

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import bf16  # noqa: E402
from xdsl.ir.affine import AffineMap  # noqa: E402

from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.compiler.kernels.registry import AIE_KERNELS  # noqa: E402
from stream.workload.node import ComputationNode  # noqa: E402
from stream.workload.tensor import Tensor  # noqa: E402

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


def _gemm_node(heads: int, maps):
    lead = (heads,) if heads else ()
    a, b = Tensor.create("a", bf16, (*lead, 32, 64)), Tensor.create("b", bf16, (*lead, 64, 16))
    out = Tensor.create("c", bf16, (*lead, 32, 16))
    return ComputationNode(type="MatMul", name="mm", inputs=(a, b), outputs=(out,), operand_mapping=maps)


@pytest.mark.parametrize(
    "node, expected",
    [
        # Gemm iterates (m, k, n).
        (
            _gemm_node(
                0,
                tuple(
                    AffineMap.from_callable(f)
                    for f in (lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n))
                ),
            ),
            [(1, 64, "k"), (2, 16, "n"), (0, 32, "m")],
        ),
        # A batched MatMul iterates (h, m, n, k): the heads lead and the kernel never sees them.
        (
            _gemm_node(
                4,
                tuple(
                    AffineMap.from_callable(f)
                    for f in (lambda h, m, n, k: (h, m, k), lambda h, m, n, k: (h, k, n), lambda h, m, n, k: (h, m, n))
                ),
            ),
            [(3, 64, "k"), (2, 16, "n"), (1, 32, "m")],
        ),
    ],
    ids=["gemm", "batched_matmul"],
)
def test_a_kernel_binds_the_library_dimensions_to_its_node_dimensions(node, expected):
    """By what each dimension does, the output's rows and columns and the contraction, not by
    where a parser happened to put it."""
    gemm = AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default", library=LIBRARY)
    assert [(position, size, d.name) for position, size, d in gemm.call_tile(node)] == expected
    assert gemm.spec.symbol == "matmul_bf16_bf16"


def test_the_object_a_kernel_links_is_named_by_the_library():
    assert AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default", library=LIBRARY).linkwith_name == "mm_32_64_16.o"


def test_a_shape_the_library_does_not_compile_is_rejected():
    with pytest.raises(ValueError, match="m=64"):
        AIE_KERNELS["gemm"](m=64, k=64, n=16, layout="default", library=LIBRARY).validate()


def test_a_kernel_without_a_library_says_so():
    with pytest.raises(ValueError, match="needs a kernel library"):
        AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default").validate()
