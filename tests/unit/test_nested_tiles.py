"""A dim tiled at two nested sizes loops twice: over the small tiles within a large one, and over the large ones."""

from __future__ import annotations

from types import SimpleNamespace

from stream.datatypes import LayerDim
from stream.workload.utils import fusion_levels

K, M, N = LayerDim("d0"), LayerDim("d1"), LayerDim("d2")


def _mapping(*tiling: tuple[LayerDim, int]) -> SimpleNamespace:
    return SimpleNamespace(fused_groups=[SimpleNamespace(intra_core_tiling=tiling)])


def test_a_k_tiled_by_64_within_512_loops_8_then_16_times():
    """K 8192 over one core in 64-wide tiles is 128 blocks: 8 tiles to a 512-wide block, 16 such blocks."""
    mapping = _mapping((K, 64), (K, 512), (N, 64), (M, 64))
    assert fusion_levels(mapping, {K: 128, N: 4, M: 8}) == [(K, 8), (K, 16), (N, 4), (M, 8)]


def test_a_dim_tiled_once_keeps_one_loop():
    assert fusion_levels(_mapping((K, 64), (M, 64)), {K: 32, M: 8}) == [(K, 32), (M, 8)]
