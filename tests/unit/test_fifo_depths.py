import pytest

pytest.importorskip("xdsl_aie", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl_aie.dialects.aie import TileOp  # noqa: E402

from stream.compiler.fifo_depths import FifoDepths, TileBudget, elem_bits, object_bytes  # noqa: E402


def tiles(*coords):
    return tuple(TileOp(c, r).result for c, r in coords)


def budget(**kwargs):
    return FifoDepths(budgets={(0, 1): TileBudget(bytes_free=1 << 20, bds_free=40), **kwargs})


def test_mem_feed_deepens():
    depths = budget().deepen((2, 2), tiles((0, 1), (0, 2)), object_bytes=8192)
    assert depths == (4, 2)


def test_non_feed_keeps_defaults():
    depths = budget().deepen((2, 2), tiles((0, 1), (0, 2)), object_bytes=8192, feed=False)
    assert depths == (2, 2)


def test_no_slack_keeps_defaults():
    policy = FifoDepths(budgets={(0, 1): TileBudget(bytes_free=1024, bds_free=40)})
    assert policy.deepen((2, 2), tiles((0, 1), (0, 2)), 8192) == (2, 2)


def test_no_descriptors_keeps_defaults():
    policy = FifoDepths(budgets={(0, 1): TileBudget(bytes_free=1 << 20, bds_free=1)})
    assert policy.deepen((2, 2), tiles((0, 1), (0, 2)), 8192) == (2, 2)


def test_budget_is_debited_across_fifos():
    policy = FifoDepths(budgets={(0, 1): TileBudget(bytes_free=70000, bds_free=40)})
    assert policy.deepen((2, 2), tiles((0, 1), (0, 2)), 16384) == (4, 2)
    assert policy.deepen((2, 2), tiles((0, 1), (0, 2)), 16384) == (2, 2)


def test_deliberate_depths_are_kept():
    assert budget().deepen((1, 1), tiles((0, 1), (0, 2)), 64) == (1, 1)
    assert budget().deepen((8, 2), tiles((0, 1), (0, 2)), 64)[0] == 8


def test_compute_consumer_deepens_producer_does_not():
    policy = FifoDepths(budgets={(0, 3): TileBudget(bytes_free=1 << 20, bds_free=40)})
    assert policy.deepen((2, 2), tiles((0, 3), (0, 3)), 1024) == (2, 4)


def test_unknown_tile_keeps_defaults():
    assert FifoDepths().deepen((2, 2), tiles((0, 1), (0, 2)), 64) == (2, 2)


def test_object_bytes():
    from xdsl.dialects.builtin import BFloat16Type

    assert object_bytes(elem_bits(BFloat16Type()), (64, 64)) == 8192
