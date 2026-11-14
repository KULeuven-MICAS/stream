"""A GEMM's precision picks how mm.cc multiplies, the library entry it is priced by, and the MAC tile it tiles in.

IRON's GEMM builds mm.cc with bfp16 multiplies only at ``Precision.MEDIUM``; at ``HIGH`` it multiplies in bf16 on
AIE2P's 4 x 8 x 8 tile. A kernel that bound one and tiled for the other would read its operands in the wrong order.
"""

from __future__ import annotations

import pytest

pytest.importorskip("aie.iron.kernels", reason="the library binds mm.cc through mlir-aie")

from stream.compiler.kernels.aie_kernel import tiled_layout  # noqa: E402
from stream.compiler.kernels.binding import resolve  # noqa: E402
from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.compiler.kernels.registry import AIE_KERNELS  # noqa: E402

DIMS = [{"name": "k", "divisor": 8}, {"name": "n", "divisor": 16}, {"name": "m", "divisor": 8}]
LIBRARY = KernelLibrary.from_dict(
    {
        "family": {"matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}}},
        "kernel": {
            "matmul_bf16_bf16": {"family": "matmul", "binding": "stream.compiler.kernels.mlir_aie:mm", "dims": DIMS},
            "matmul_bf16_bf16_high": {
                "family": "matmul",
                "binding": "stream.compiler.kernels.mlir_aie:mm",
                "dims": DIMS,
                "mac": {"m": 4, "k": 8, "n": 8},
                "options": {"bfp16": False},
            },
        },
    }
)


def _binding(kernel):
    spec = kernel.spec
    return resolve({"binding": spec.binding, "args": {**kernel.call_shape(), **spec.options, "npu": "npu2"}})


@pytest.mark.parametrize(
    "precision, entry, mac, bfp16",
    [("medium", "matmul_bf16_bf16", (8, 8, 8), True), ("high", "matmul_bf16_bf16_high", (4, 8, 8), False)],
)
def test_a_gemm_builds_and_tiles_as_its_precision_says(precision, entry, mac, bfp16):
    """``medium`` is the library's plain entry, mm.cc's bfp16 build on the family's 8 x 8 x 8 tile; ``high`` the entry
    built without bfp16, on 4 x 8 x 8,
    so A, 32 x 64, is tiled in tiles of 4 x 8 rather than 8 x 8."""
    kernel = AIE_KERNELS["gemm"](m=32, k=64, n=32, layout="default", precision=precision, library=LIBRARY)
    assert kernel.spec.symbol == entry
    assert (kernel.mac["m"], kernel.mac["k"], kernel.mac["n"]) == mac
    assert ("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" in _binding(kernel).compile_flags) is bfp16
    assert kernel.operand_layouts()[0] == tiled_layout(32, 64, mac[0], mac[1])
