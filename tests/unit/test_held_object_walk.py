"""A memory tile forwarding the object it holds walks all of it, as mlir-aie's link verifier requires."""

from math import prod

import pytest

pytest.importorskip("xdsl_aie", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import SymbolRefAttr, bf16  # noqa: E402
from xdsl_aie.dialects.aie import DeviceOp, ObjectFifoLinkOp, ObjectFifoOp, TileOp  # noqa: E402

from stream.compiler.transforms.convert_stream_to_aie import walk_held_object  # noqa: E402

# A kernel tile's blocked walk: rows of 8x8 blocks, each block row by row.
BLOCKED = ([8, 8, 8, 8], [512, 8, 64, 1])


def _forward(held: list[int], outputs: int = 1) -> tuple[DeviceOp, list[ObjectFifoOp]]:
    """A memory tile holding ``held`` and handing it to cores as 64x64 kernel tiles through ``outputs`` fifos."""
    shim, mem, cores = TileOp(0, 0), TileOp(0, 1), [TileOp(0, 2 + i) for i in range(outputs)]
    slab = ObjectFifoOp.from_referenced_type(shim, [mem], "slab", 1, bf16, held)
    tiles = [
        ObjectFifoOp.from_referenced_type(mem, [core], f"tile_{i}", 2, bf16, [64, 64]) for i, core in enumerate(cores)
    ]
    step = prod(held) // outputs
    link = ObjectFifoLinkOp(
        [SymbolRefAttr("slab")],
        [SymbolRefAttr(t.sym_name) for t in tiles],
        [],
        [i * step for i in range(outputs)] if outputs > 1 else [],
    )
    return DeviceOp(0, [shim, mem, *cores, slab, *tiles, link]), tiles


def test_a_slab_handed_out_tile_by_tile_walks_the_whole_slab():
    """A column's weights, 4x8 kernel tiles, leave in one descriptor: the walk steps on over all 32."""
    device, (tile,) = _forward([4, 8, 64, 64])
    sizes, strides = walk_held_object(device, tile, *BLOCKED)
    assert (sizes, strides) == ([256, 8, 8, 8], [512, 8, 64, 1])
    assert prod(sizes) == 4 * 8 * 64 * 64


def test_a_walk_that_does_not_run_on_steps_over_the_tiles_with_a_dimension_of_its_own():
    device, (tile,) = _forward([2, 64, 64])
    assert walk_held_object(device, tile, [64, 64], [1, 64]) == ([2, 64, 64], [4096, 1, 64])


def test_a_tile_forwarding_one_kernel_tile_keeps_its_walk():
    device, (tile,) = _forward([64, 64])
    assert walk_held_object(device, tile, *BLOCKED) == BLOCKED


def test_a_distribute_keeps_its_walk():
    """Each output of a distribute sends its own slot, which the walk already covers."""
    device, tiles = _forward([2, 64, 64], outputs=2)
    assert walk_held_object(device, tiles[0], *BLOCKED) == BLOCKED


def test_a_fifo_no_link_fills_keeps_its_walk():
    mem, core = TileOp(0, 1), TileOp(0, 2)
    tile = ObjectFifoOp.from_referenced_type(mem, [core], "tile", 2, bf16, [64, 64])
    assert walk_held_object(DeviceOp(0, [mem, core, tile]), tile, *BLOCKED) == BLOCKED
