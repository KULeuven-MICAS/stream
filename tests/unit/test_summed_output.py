"""A GEMM summing its K tiles in float accumulators sends each output tile on once its sum is done: the core holds one
accumulator per tile it sums at once and a single bf16 output buffer it sends them through."""

from __future__ import annotations

from types import SimpleNamespace

from stream.opt.allocation.constraint_optimization.space import DecisionSpace

C = "c"


def _space(tiles: int) -> SimpleNamespace:
    """An output held to stop 1, its window ``tiles`` tiles wide, summed in an accumulator."""
    space = SimpleNamespace(
        tiles_needed_levels={(C, 1): tiles, (C, -1): 2},
        rotation_levels={(C, 1): True},
        kernel_states=set(),
        summed_outputs={C},
    )
    space.summed_tiles = lambda t, stop: DecisionSpace.summed_tiles(space, t, stop)
    return space


def test_a_summed_output_tile_is_held_once():
    """One 64x64 output tile at a time: one bf16 buffer, not two, the accumulator taking the next sum."""
    assert DecisionSpace.resident_tiles(_space(1), C, 1) == 1
    assert DecisionSpace.resident_tiles(_space(1), C, -1) == 1


def test_a_core_summing_eight_output_tiles_holds_eight_accumulators_and_one_buffer():
    """A fused down projection holds 8 output tiles across the hidden loop: 8 float sums, one bf16 output buffer."""
    space = _space(8)
    assert DecisionSpace.resident_tiles(space, C, 1) == 1
    assert DecisionSpace.summed_tiles(space, C, 1) == 8
