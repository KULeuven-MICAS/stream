"""A transfer that fans out to readers splitting it differently lowers the same way every run."""

from __future__ import annotations

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import StringAttr, bf16  # noqa: E402
from xdsl.ir import Block  # noqa: E402

from stream.compiler.dialects.stream import (  # noqa: E402
    ChannelOp,
    PullOp,
    PushOp,
    StrensorSpace,
    StrensorSpaceAttr,
    StrensorType,
    StrensorVar,
    StrensorVarType,
)
from stream.compiler.transforms.aie_convert_ofs import transfer_endpoints  # noqa: E402
from stream.datatypes import LayerDim  # noqa: E402

M, N = LayerDim("d2"), LayerDim("d1")


def strensor(column: StrensorVarType, cores: list[str]) -> StrensorType:
    space = StrensorSpace(
        (
            StrensorVar(column, 2, N),
            StrensorVar(StrensorVarType.SPATIAL, 4, M),
            StrensorVar(StrensorVarType.KERNEL, 32, M),
        )
    )
    return StrensorType(bf16, space, [StringAttr(core) for core in cores])


def point(*coordinates: tuple[int, LayerDim]) -> StrensorSpaceAttr:
    return StrensorSpaceAttr(StrensorSpace(tuple(StrensorVar(StrensorVarType.POINT, i, d) for i, d in coordinates)))


@pytest.mark.parametrize("wide_first", [True, False])
def test_a_fanned_out_transfer_ends_at_the_reader_splitting_the_most_dimensions(wide_first: bool):
    wide = strensor(StrensorVarType.SPATIAL, ["tile_1_2"])
    narrow = strensor(StrensorVarType.TEMPORAL, ["tile_0_2"])
    block = Block(arg_types=[strensor(StrensorVarType.SPATIAL, ["tile_4_1", "tile_7_1"])])
    channel = ChannelOp()
    push = PushOp(block.args[0], channel)
    pulls = [PullOp(wide, channel, point((0, N), (0, M))), PullOp(narrow, channel, point((0, M)))]
    block.add_ops([channel, push, *(pulls if wide_first else pulls[::-1])])
    assert transfer_endpoints(push) == (wide, wide)
