"""A memory tile may hold an operand past its reader only as one replayed whole."""

from __future__ import annotations

import pytest

from stream.compiler.dialects.stream import StrensorSpace, StrensorVar, StrensorVarType
from stream.datatypes import LayerDim
from stream.opt.allocation.constraint_optimization.families.reuse import (
    replay_unexpressible_levels,
    replayed_refilled_levels,
)

Q, KEY = LayerDim("d2"), LayerDim("d1")


def test_the_key_operand_stages_whole_and_replays_over_the_query():
    forbidden = replay_unexpressible_levels([True, False], read_levels=2)
    assert (1, -1) not in forbidden and (1, 0) not in forbidden


def test_a_replay_loop_inside_the_object_is_refused():
    assert (1, -1) in replay_unexpressible_levels([False, True], read_levels=2)


def test_a_slab_held_across_an_inner_loop_is_refilled_per_outer_trip():
    """Loops k, m, n innermost first: the weight's column slab, all of k, held at stop 1, replays over m and refills
    per n."""
    forbidden = replay_unexpressible_levels([True, False, True], read_levels=3)
    assert (1, -1) not in forbidden and (1, 0) not in forbidden
    assert (2, -1) in forbidden


def test_a_block_staged_inside_an_outer_loop_is_refetched_per_trip():
    """Loops k, m innermost first, the operand indexed by k only: its k block is held at stop 0 and fetched again per
    trip of m, as a core's would be."""
    assert (0, -1) not in replay_unexpressible_levels([True, False], read_levels=2)


def test_a_distributed_slab_refilled_per_outer_trip_is_refused():
    """Each reader takes its own slice of a distributed object, one tile per firing: a refilled object holding several
    tiles per reader is refused, a whole-run one and a single tile kept."""
    forbidden = replay_unexpressible_levels([True, False, True], read_levels=3, distributed=True)
    assert (0, -1) in forbidden and (1, -1) in forbidden
    assert (1, 0) not in forbidden
    assert replay_unexpressible_levels([True, False], read_levels=2, distributed=True) == [(0, -1)]


def test_an_object_refilled_per_row_block_and_replayed_over_the_column_blocks():
    """Loops k, n, m innermost first, 8, 8 and 2 trips: the operand indexed by k and m, held at stop 1 across n and
    refilled per m, is replayed to a reader holding at most one k block on each trip of n."""
    assert replayed_refilled_levels([8, 8, 2], [True, False, True], read_levels=3) == [(1, -1), (1, 0)]


def test_an_object_held_to_the_last_loop_or_read_on_new_data_is_no_refilled_replay():
    assert replayed_refilled_levels([8, 8, 1], [True, False, True], read_levels=3) == []
    assert replayed_refilled_levels([8, 1, 2], [True, False, True], read_levels=3) == []


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
