"""A memory tile may hold an operand past its reader only as one replayed whole."""

from __future__ import annotations

import pytest

from stream.compiler.dialects.stream import StrensorSpace, StrensorVar, StrensorVarType
from stream.datatypes import LayerDim
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    replay_unexpressible_levels,
)

Q, KEY = LayerDim("d2"), LayerDim("d1")


def test_the_key_operand_stages_whole_and_replays_over_the_query():
    forbidden = replay_unexpressible_levels([True, False], read_levels=2)
    assert (1, -1) not in forbidden and (1, 0) not in forbidden
    assert (0, -1) in forbidden


def test_a_replay_loop_inside_the_object_is_refused():
    assert (1, -1) in replay_unexpressible_levels([False, True], read_levels=2)


def test_a_partial_window_stage_is_refused():
    forbidden = replay_unexpressible_levels([True, True, False], read_levels=3)
    assert (1, -1) in forbidden and (1, 0) in forbidden
    assert (2, -1) not in forbidden


def _strensor_space(reuse_index, *vars):
    return type(
        "S",
        (),
        {
            "ssis": type("A", (), {"data": StrensorSpace(tuple(vars))})(),
            "reuse_index": type("A", (), {"data": reuse_index})(),
        },
    )()


def _var(kind, size, dim):
    return StrensorVar(kind, size, dim)


def test_replay_count_is_the_producer_window_loops_the_consumer_refires_on():
    aie_convert_ofs = pytest.importorskip(
        "stream.compiler.transforms.aie_convert_ofs",
        reason="the AIE dialects are a separate install, via stream-setup-aie",
    )
    kernel = (_var(StrensorVarType.KERNEL, 64, KEY),)
    producer = _strensor_space(
        3,
        _var(StrensorVarType.TEMPORAL, 4, Q),
        _var(StrensorVarType.TEMPORAL, 32, KEY),
        *kernel,
    )
    consumer = _strensor_space(
        1,
        _var(StrensorVarType.TEMPORAL, 4, Q),
        _var(StrensorVarType.TEMPORAL, 32, KEY),
        *kernel,
    )
    assert aie_convert_ofs.ChannelToObjectFifoPass.replay_count(producer, consumer) == 4
    assert aie_convert_ofs.ChannelToObjectFifoPass.replay_count(consumer, consumer) == 1
