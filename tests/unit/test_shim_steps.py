import pytest

pytest.importorskip("xdsl_aie", reason="the AIE dialects are a separate install, via stream-setup-aie")

from stream.compiler.transforms.aie_convert_ofs import (  # noqa: E402
    SHIM_STEP_WORDS,
    WORD_BITS,
    Stride,
    StrideSet,
)

BF16_STEP = SHIM_STEP_WORDS * WORD_BITS // 16
ROWS = 256 * 14336
"""Two blocks of 256 rows of a (512, 14336) output, 64 columns at a time over 28 column blocks: the block step is
further than a bf16 shim descriptor steps."""


@pytest.mark.parametrize(
    "strides, transfers",
    [
        ((Stride(64, 1, 0), Stride(256, 14336, 0), Stride(28, 512, 1), Stride(2, ROWS, 28)), 2),
        ((Stride(64, 1, 0), Stride(2, ROWS, 0), Stride(28, 512, 2)), 56),
    ],
)
def test_the_dimensions_from_one_a_descriptor_cannot_step_become_separate_transfers(strides, transfers):
    legal = StrideSet(strides).legalize(BF16_STEP)
    hardware, software = legal.strides[:4], [var for var in legal.strides[4:] if var.size > 1]
    first = next(i for i, var in enumerate(strides) if var.stride > BF16_STEP)
    assert all(var.stride <= BF16_STEP for var in hardware if var.size > 1)
    assert software == list(strides[first:])
    count = 1
    for var in software:
        count *= var.size
    assert count == transfers
