"""A memory tile holds a window an outer loop moves on as whole objects, so it may hold a slab of several tiles once,
the next window waiting for the last read of this one; a core keeps a rolling fifo of single tiles instead."""

from __future__ import annotations

from types import SimpleNamespace

from stream.opt.allocation.constraint_optimization.space import DecisionSpace

SLAB = "slab"


def _space(kind: str, tiles: int = 32) -> SimpleNamespace:
    """A tensor at stop 1 whose window holds ``tiles`` kernel tiles, moved on by an outer loop, in a ``kind`` memory."""
    space = SimpleNamespace(
        tiles_needed_levels={(SLAB, 1): tiles},
        rotation_levels={(SLAB, 1): True},
        tensors_to_optimize_reuse_for=[SLAB],
        sole_outputs={SLAB},
        kernel_states=set(),
        summed_outputs=set(),
        candidate_memories=lambda t: [SimpleNamespace(kind=kind)],
    )
    space.staged_whole = lambda t: DecisionSpace.staged_whole(space, t)
    return space


def test_a_memory_tile_may_hold_a_slab_once():
    """A 32-tile slab (64 x 2048 of an activation): 64 tiles double buffered, 32 held once."""
    space = _space("memory")
    assert DecisionSpace.may_single_buffer(space, SLAB, 1)
    assert DecisionSpace.resident_tiles(space, SLAB, 1, staged=True) == 64
    assert DecisionSpace.resident_tiles(space, SLAB, 1, single=True, staged=True) == 32


def test_a_core_window_of_several_tiles_is_not_held_once():
    assert not DecisionSpace.may_single_buffer(_space("compute"), SLAB, 1)
