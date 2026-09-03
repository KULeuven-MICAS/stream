from stream.mapping.chain_placement import (
    column_budgets,
    row_counts,
    widest_columns,
)


def test_equal_width_beats_a_wider_softmax():
    # scores / softmax / context per-call cycles: the wider softmax halves its work but
    # pays the tiled handover, a wash the bottleneck rule refuses to spend a row on.
    assert row_counts([1730, 4400, 1536], 4) == (1, 1, 1)


def test_a_dominant_final_layer_widens_free_of_the_handover_tax():
    assert row_counts([100, 1000], 4) == (1, 3)


def test_a_dominant_middle_layer_does_not_widen():
    # Its extra row halves the work and the MAC-tiled handover doubles it back.
    assert row_counts([1730, 4400 * 2, 1536], 4) == (1, 1, 1)


def test_widest_columns_honours_granularity():
    assert widest_columns(512, 64, 1, 8) == 8
    assert widest_columns(256, 64, 1, 8) == 4
    assert widest_columns(512, 64, 2, 8) == 4


def test_column_budgets_reproduce_the_swiglu_split():
    gemm, elt = 1730.0, 128.0
    assert column_budgets([gemm, gemm, elt, elt, gemm], 8) == (2, 2, 1, 1, 2)
