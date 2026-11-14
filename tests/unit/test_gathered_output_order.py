"""The runtime sequence writes a gathered output off chip in the order its memory tile sends it: one round at a time,
each round one output tile of every core, while each core holds several output tiles, along another dimension or
along the one the cores split, inside or outside the cores' split."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

pytest.importorskip("xdsl_aie", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, bf16  # noqa: E402
from xdsl.ir import Block, Region  # noqa: E402
from xdsl.pattern_rewriter import PatternRewriteWalker  # noqa: E402
from xdsl_aie.dialects.aie import CoreOp, DeviceOp, EndOp, RuntimeSequenceOp, TileOp  # noqa: E402
from xdsl_aie.dialects.aiex import DmaConfigureTaskForOp  # noqa: E402

from stream.compiler.dialects.stream import (  # noqa: E402
    ChannelOp,
    PullOp,
    PushOp,
    StrensorSpace,
    StrensorSpaceAttr,
    StrensorType,
    StrensorVar,
    StrensorVarType,
    YieldOp,
)
from stream.compiler.transforms.aie_convert_ofs import TransferToRuntimeSequence  # noqa: E402
from stream.compiler.transforms.aie_tiles import COMPUTE, MEMORY, OFFCHIP, AIETileLocation  # noqa: E402
from stream.datatypes import LayerDim  # noqa: E402

M, N = LayerDim("d0"), LayerDim("d2")
ROWS, HELD, TILE = 4, 8, 32
"""Four cores of a column split the rows; each holds eight 32x32 output tiles."""
CORES = [f"tile_0_{2 + r}" for r in range(ROWS)]
TILES = {
    "dram": AIETileLocation(OFFCHIP, 0, 0),
    "tile_0_1": AIETileLocation(MEMORY, 0, 1),
    **{core: AIETileLocation(COMPUTE, 0, 2 + r) for r, core in enumerate(CORES)},
}
COLUMNS, ROWS_OUTSIDE, ROWS_INSIDE = "columns", "rows outside the cores", "rows inside the cores"
"""Where each core's held tiles lie: along the columns, or along the rows, as the outer or the inner of the two row
loops."""


def _var(kind: StrensorVarType, size: int, dim: LayerDim) -> StrensorVar:
    return StrensorVar(kind, size, dim)


def _strensor(held: str, join: StrensorVarType, places: list[str]) -> StrensorType:
    tiles, cores = _var(StrensorVarType.TEMPORAL, HELD, N if held == COLUMNS else M), _var(join, ROWS, M)
    loops = (cores, tiles) if held == ROWS_INSIDE else (tiles, cores)
    space = StrensorSpace((*loops, _var(StrensorVarType.KERNEL, TILE, M), _var(StrensorVarType.KERNEL, TILE, N)))
    return StrensorType(bf16, space, [StringAttr(p) for p in places], reuse_index=len(space.vars))


def _index(r: int) -> StrensorSpaceAttr:
    return StrensorSpaceAttr(StrensorSpace((_var(StrensorVarType.POINT, r, M),)))


def _width(held: str) -> int:
    return HELD * TILE if held == COLUMNS else TILE


def _lowered(held: str) -> list[DmaConfigureTaskForOp]:
    """The output's transfer chain, cores to memory tile to DRAM, with its runtime pull lowered."""
    height = ROWS * TILE * (1 if held == COLUMNS else HELD)
    dram = StrensorType(
        bf16,
        StrensorSpace((_var(StrensorVarType.CONSTANT, height, M), _var(StrensorVarType.CONSTANT, _width(held), N))),
        [StringAttr("dram")],
    )
    gathered, off_chip = ChannelOp(), ChannelOp()
    produced = _strensor(held, StrensorVarType.SPATIAL, CORES)
    cores = []
    for r in range(ROWS):
        body = Block(arg_types=[produced])
        body.add_ops([PushOp(body.args[0], gathered, _index(r)), EndOp()])
        cores.append(CoreOp(None, TileOp(0, 2 + r), Region(body)))
    staged = PullOp(_strensor(held, StrensorVarType.TEMPORAL, ["tile_0_1"]), gathered)
    relay = CoreOp(None, TileOp(0, 1), Region(Block([staged, PushOp(staged.output, off_chip), EndOp()])))
    sequence = Block(arg_types=[dram])
    pull = PullOp(dram, off_chip)
    pull.attributes["of"] = ArrayAttr([StringAttr("of_out")])
    sequence.add_ops([pull, YieldOp(pull.output)])
    device = DeviceOp(0, [gathered, off_chip, *cores, relay, RuntimeSequenceOp(Region(sequence))])
    PatternRewriteWalker(TransferToRuntimeSequence(TILES)).rewrite_module(ModuleOp([device]))
    return [op for op in device.walk() if isinstance(op, DmaConfigureTaskForOp)]


def _first_row(held: str, j: int, r: int) -> int:
    """The output row tile ``j`` of core ``r`` starts at."""
    if held == COLUMNS:
        return r * TILE
    if held == ROWS_OUTSIDE:
        return (j * ROWS + r) * TILE
    return (r * HELD + j) * TILE


@pytest.mark.parametrize("held", [COLUMNS, ROWS_OUTSIDE, ROWS_INSIDE])
def test_a_gathered_output_lands_round_by_round_as_its_memory_tile_sends_it(held):
    """Round ``j`` holds output tile ``j`` of every core, core by core, each tile where the space nests it: along the
    columns, tile ``j`` of core ``r`` is rows ``r * 32`` on, columns ``j * 32`` on."""
    width = _width(held)
    column = TILE if held == COLUMNS else 0
    sent = [
        (_first_row(held, j, r) + row) * width + j * column + col
        for j, r, row, col in product(range(HELD), range(ROWS), range(TILE), range(TILE))
    ]
    walked = []
    for task in _lowered(held):
        (bd, _) = task.body.block.ops
        sizes, strides = bd.static_sizes.get_values(), bd.static_strides.get_values()
        offset = bd.static_offset.value.data
        walked += [
            offset + sum(i * s for i, s in zip(index, strides, strict=True)) for index in product(*map(range, sizes))
        ]
    assert np.array_equal(walked, sent)
