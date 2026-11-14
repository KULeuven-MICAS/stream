import pytest

pytest.importorskip("xdsl_aie", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import ModuleOp, StringAttr, bf16  # noqa: E402
from xdsl_aie.dialects.aie import (  # noqa: E402
    BDDimLayout,
    BDDimLayoutArray,
    BDDimLayoutArrayArray,
    BDDimLayoutArrayArrayAttr,
    DeviceOp,
    ObjectFifoOp,
    TileOp,
)

from stream.compiler.transforms.aie_convert_ofs import RealizeLinks  # noqa: E402

SLOT = 16 * 32 * 32
"""One reader's slot of a (4 rows, 16 columns) block of 32x32 tiles, a row of tiles a reader."""
ROWS_ACROSS_TILES = ((32, 32), (16, 1024), (32, 1))
"""A row of every tile in turn: a widened run off chip scattered into the tiles of one slot."""


def _staged(walk):
    shim, mem = TileOp(0, 0), TileOp(0, 1)
    layout = BDDimLayoutArrayArrayAttr(BDDimLayoutArrayArray((BDDimLayoutArray(tuple(map(BDDimLayout, walk))),)))
    fifo = ObjectFifoOp.from_referenced_type(shim, [mem], "staged", 1, bf16, [4, 16, 32, 32])
    fifo.dimensionsFromStreamPerConsumer = layout
    device = DeviceOp(0, [shim, mem, fifo])
    ModuleOp([device])
    return device, fifo


@pytest.mark.parametrize("walk", [((4, SLOT), *ROWS_ACROSS_TILES), ROWS_ACROSS_TILES])
def test_a_distributed_object_is_filled_by_a_walk_over_one_slot(walk):
    device, fifo = _staged(walk)
    RealizeLinks.fill_slot_by_slot(device, StringAttr("staged"), 4)
    assert [tuple(dim) for dim in fifo.dimensionsFromStreamPerConsumer.data[0]] == list(ROWS_ACROSS_TILES)


def test_a_walk_crossing_the_slots_is_refused():
    device, _ = _staged(((32, 32), (4, SLOT), (16, 1024), (32, 1)))
    with pytest.raises(NotImplementedError, match="slots"):
        RealizeLinks.fill_slot_by_slot(device, StringAttr("staged"), 4)
