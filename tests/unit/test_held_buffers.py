"""How many buffers a tile holds of a tensor whose window a loop outside its reuse moves on."""

from __future__ import annotations

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl.context import Context  # noqa: E402
from xdsl.dialects.builtin import StringAttr, bf16  # noqa: E402
from xdsl.parser import Parser  # noqa: E402

from stream.compiler.dialects.stream import (  # noqa: E402
    Stream,
    StrensorSpace,
    StrensorType,
    StrensorVar,
    StrensorVarType,
)
from stream.compiler.transforms.aie_convert_ofs import ChannelToObjectFifoPass  # noqa: E402
from stream.datatypes import LayerDim  # noqa: E402

HEAD, QUERY, KEY = LayerDim(0), LayerDim(1), LayerDim(2)


def key(buffers: int = 2, heads: int = 4) -> StrensorType:
    """A head's whole key on a core, reused over the query loop and moved on by the heads loop."""
    space = StrensorSpace(
        (
            StrensorVar(StrensorVarType.TEMPORAL, heads, HEAD),
            StrensorVar(StrensorVarType.TEMPORAL, 8, QUERY),
            StrensorVar(StrensorVarType.KERNEL, 64, KEY),
        )
    )
    return StrensorType(bf16, space, [StringAttr("tile_0_2")], reuse_index=2, buffers=buffers)


@pytest.mark.parametrize("buffers", [1, 2])
def test_a_moving_window_is_held_in_the_buffers_the_solve_chose(buffers):
    assert ChannelToObjectFifoPass.held_count(key(buffers)) == buffers


def test_a_window_nothing_moves_on_is_held_once():
    space = StrensorSpace(
        (StrensorVar(StrensorVarType.TEMPORAL, 8, QUERY), StrensorVar(StrensorVarType.KERNEL, 64, KEY))
    )
    assert ChannelToObjectFifoPass.held_count(StrensorType(bf16, space, [StringAttr("tile_0_2")], reuse_index=2)) == 1


@pytest.mark.parametrize("buffers", [1, 2])
def test_the_buffer_count_survives_printing(buffers):
    context = Context()
    context.load_dialect(Stream)
    assert Parser(context, str(key(buffers))).parse_attribute() == key(buffers)
