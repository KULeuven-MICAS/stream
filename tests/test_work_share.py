"""The share of a node's work each core does, against what the hardware was traced doing."""

from math import ceil

import pytest

# Traced on NPU2 Strix, seq 512 / 32 heads, softmax row across all eight columns, both
# compiled query blocks. 260912_kernel_manifest/results/columns512.jsonl. Computed calls
# per column, divided by the 32 heads.
TRACED = {
    32: [6, 6, 8, 8, 10, 10, 12, 12],
    64: [1, 2, 3, 4, 5, 6, 7, 8],
}


def attended(index, width, steps, query_tile, key_tile):
    """Key blocks the core at ``index`` attends, by the rule work_share applies."""
    return sum(ceil((b + 1) * query_tile / key_tile) for b in range(index, width * steps, width))


@pytest.mark.parametrize("query_tile", [32, 64])
def test_the_causal_share_matches_what_the_columns_were_traced_computing(query_tile):
    width, key_tile, seq = 8, 64, 512
    steps = seq // query_tile // width
    got = [attended(i, width, steps, query_tile, key_tile) for i in range(width)]
    assert got == TRACED[query_tile]


def test_both_blocks_do_the_same_work_and_only_its_spread_differs():
    """Which is the whole point: a model that prices the total cannot tell them apart, and
    the dispatch waits for the busiest column."""
    work, imbalance = {}, {}
    for query_tile in (32, 64):
        steps = 512 // query_tile // 8
        per_core = [attended(i, 8, steps, query_tile, 64) for i in range(8)]
        # An attended unit is one key block of key_tile rows by query_tile query rows.
        work[query_tile] = sum(per_core) * query_tile * 64
        imbalance[query_tile] = max(per_core) / (sum(per_core) / 8)

    assert work[32] == work[64] == 147456
    assert round(imbalance[64], 3) == round(8 / 4.5, 3) == 1.778
    assert round(imbalance[32], 3) == 1.333
    assert imbalance[64] / imbalance[32] > 1.3
