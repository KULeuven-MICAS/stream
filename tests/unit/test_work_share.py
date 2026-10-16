"""The share of a causal node's work each core does, against what the hardware was traced doing."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from stream.compiler.kernels.flash import CausalGemmKernel  # noqa: E402
from stream.compiler.kernels.library import KernelLibrary  # noqa: E402

LIBRARY = KernelLibrary.load(Path(__file__).parents[2] / "stream/inputs/aie/kernels/aie2p.toml")
WIDTH, SEQ, KEY_TILE = 8, 512, 64
TRACED = {
    32: [6, 6, 8, 8, 10, 10, 12, 12],
    64: [1, 2, 3, 4, 5, 6, 7, 8],
}


def shares(query_tile: int) -> list[float]:
    kernel = CausalGemmKernel(m=query_tile, k=KEY_TILE, n=KEY_TILE, layout="default", library=LIBRARY)
    steps = SEQ // query_tile // WIDTH
    return [kernel.work_share(index, WIDTH, steps) for index in range(WIDTH)]


@pytest.mark.parametrize("query_tile", [32, 64])
def test_each_core_does_the_key_blocks_its_column_was_traced_attending(query_tile: int):
    steps = SEQ // query_tile // WIDTH
    total = WIDTH * steps * (SEQ // KEY_TILE)
    assert shares(query_tile) == pytest.approx([blocks / total for blocks in TRACED[query_tile]])


def test_the_finer_query_block_spreads_the_causal_work_more_evenly():
    imbalance = {}
    for query_tile in (32, 64):
        per_core = shares(query_tile)
        imbalance[query_tile] = max(per_core) / (sum(per_core) / WIDTH)
    assert imbalance[64] == pytest.approx(16 / 9)
    assert imbalance[32] == pytest.approx(4 / 3)
