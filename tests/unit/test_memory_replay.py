"""A memory tile may hold an operand past its reader only as one replayed whole.

The DMA re-sends the staged object with a single start-queue repeat, so the extra
residency exists exactly for whole-window stages whose replay loops all sit outside
the object. Everything else stays at the reader's stop, as before.
"""

from __future__ import annotations

from stream.compiler.dialects.stream import StrensorSpace, StrensorVar, StrensorVarType
from stream.compiler.transforms.aie_convert_ofs import ChannelToObjectFifoPass
from stream.datatypes import LayerDim
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    replay_unexpressible_levels,
)

Q, KEY = LayerDim("d2"), LayerDim("d1")


def test_the_key_operand_stages_whole_and_replays_over_the_query():
    # K under flash: key blocks stream (relevant, inner), the query loop re-reads
    # them (irrelevant, outer). Only the whole-window stop escapes the reader's.
    forbidden = replay_unexpressible_levels([True, False], read_levels=2)
    assert (1, -1) not in forbidden and (1, 0) not in forbidden
    assert (0, -1) in forbidden


def test_a_replay_loop_inside_the_object_is_refused():
    # An irrelevant loop under a relevant one would need the repeat inside the
    # object; one start-queue repeat cannot say that, whole window or not.
    assert (1, -1) in replay_unexpressible_levels([False, True], read_levels=2)


def test_a_partial_window_stage_is_refused():
    # A mid-window pool unrolls per tile in the lowering (the k=1 SwiGLU 16-tile
    # pool made a memory tile exceed its 48 BD blocks), so only the top offers.
    forbidden = replay_unexpressible_levels([True, True, False], read_levels=3)
    assert (1, -1) in forbidden and (1, 0) in forbidden
    assert (2, -1) not in forbidden


def _strensor_space(reuse_index, *vars):
    return type("S", (), {
        "ssis": type("A", (), {"data": StrensorSpace(tuple(vars))})(),
        "reuse_index": type("A", (), {"data": reuse_index})(),
    })()


def _var(kind, size, dim):
    return StrensorVar(kind, size, dim)


def test_replay_count_is_the_producer_window_loops_the_consumer_refires_on():
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
    assert ChannelToObjectFifoPass.replay_count(producer, consumer) == 4
    assert ChannelToObjectFifoPass.replay_count(consumer, consumer) == 1
