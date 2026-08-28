"""Trace tile selection: a core needs a south port out of its column, a mem tile does not."""

from xdsl.dialects.builtin import MemRefType, ModuleOp, bf16
from xdsl_aie.dialects.aie import ObjectFIFO, ObjectFifoOp, TileOp

from stream.compiler.transforms.aie_add_tracing_script import _SOUTH_PORTS, _blocked_columns


def _fifo(name: str, producer: TileOp, consumer: TileOp) -> ObjectFifoOp:
    return ObjectFifoOp(producer.result, [consumer.result], 2, ObjectFIFO([MemRefType(bf16, [64, 64])]), name)


def _column(tiles: dict[tuple[int, int], TileOp], col: int, streams_down: int) -> list[ObjectFifoOp]:
    """``streams_down`` cores sending to the column's mem tile; the rest hand off sideways."""
    ops = []
    for row in range(2, 6):
        target = tiles[(col, 1)] if row - 2 < streams_down else tiles[(col + 1, row)]
        ops.append(_fifo(f"of_{col}_{row}", tiles[(col, row)], target))
    return ops


def _module(*columns: int) -> ModuleOp:
    tiles = {(c, r): TileOp(c, r) for c in range(len(columns) + 1) for r in range(1, 6)}
    ops: list = list(tiles.values())
    for col, streams_down in enumerate(columns):
        ops += _column(tiles, col, streams_down)
    return ModuleOp(ops)


def test_column_that_sends_every_core_down_is_blocked():
    # The layer-by-layer shape: each core streams to its own mem tile, taking every south port.
    assert _blocked_columns(_module(_SOUTH_PORTS)) == {0}


def test_column_that_hands_off_sideways_is_free():
    # The fused shape: cores feed the next stage's column, so nothing descends.
    assert _blocked_columns(_module(0)) == set()


def test_a_spare_south_port_leaves_the_column_traceable():
    assert _blocked_columns(_module(_SOUTH_PORTS - 1)) == set()


def test_a_stream_counts_against_the_column_it_lands_in():
    # Cores in column 1 joining column 0's mem tile descend in column 0.
    tiles = {(c, r): TileOp(c, r) for c in (0, 1) for r in range(1, 6)}
    ops: list = list(tiles.values())
    ops += [_fifo(f"of_{r}", tiles[(1, r)], tiles[(0, 1)]) for r in range(2, 6)]
    assert 0 in _blocked_columns(ModuleOp(ops))
