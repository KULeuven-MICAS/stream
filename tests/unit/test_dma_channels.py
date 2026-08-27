"""What a transfer costs the tile it lands on.

The object-fifo lowering spends a DMA channel per fifo it splits and none for the fifos it
leaves in memory two tiles already share. A model that counts either of those wrong is not
caught by any test the solver runs: it simply hands aiecc a design whose channels do not
fit, and the failure arrives as ``number of input DMA channel exceeded`` with no mapping
back to the choice that caused it.
"""

from __future__ import annotations

import pytest

from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.context import AIE2Constraints


def _core(core_id: int, kind: str, col: int | None, row: int | None, namespace: str = "aie2") -> Core:
    core = Core.__new__(Core)
    core.id = core_id
    core.core_type = f"{namespace}.{kind}"
    core.type = kind
    core.col_id = col
    core.row_id = row
    return core


@pytest.fixture
def aie2():
    return AIE2Constraints(offchip_core_id=None)


def test_a_compute_tile_has_the_two_dma_channels_the_hardware_gives_it(aie2):
    """``AIE2TargetModel`` answers 2 for ``WireBundle::DMA`` on a compute tile, 6 on a
    memory tile. Modelling more is what lets an infeasible design reach aiecc."""
    assert aie2.get_max_dma_channels(_core(0, "compute", 0, 2)) == 2
    assert aie2.get_max_dma_channels(_core(1, "memory", 0, 1)) == 6


@pytest.mark.parametrize(
    "one, other, shared",
    [
        ((0, 2), (0, 3), True),  # north
        ((0, 3), (0, 2), True),  # south
        ((0, 2), (1, 2), True),  # east
        ((0, 2), (0, 4), False),  # two rows apart
        ((0, 2), (1, 3), False),  # diagonal
        ((0, 2), (0, 2), False),  # itself is not a transfer
    ],
)
def test_a_compute_tile_reaches_the_memory_of_the_tiles_around_it(aie2, one, other, shared):
    """``isLegalMemAffinity``: north, south, east and west, and nothing further."""
    assert aie2.shares_memory(_core(0, "compute", *one), _core(1, "compute", *other)) is shared


def test_a_memory_tile_is_never_on_the_shared_path(aie2):
    """A core reaches a memory tile over a stream, so that transfer costs a channel."""
    assert aie2.shares_memory(_core(0, "compute", 0, 2), _core(1, "memory", 0, 1)) is False


def test_a_core_with_no_position_is_not_assumed_adjacent(aie2):
    """An accelerator description that leaves the grid out must not read as neighbours."""
    assert aie2.shares_memory(_core(0, "compute", None, None), _core(1, "compute", 0, 2)) is False


class _Plan:
    """Just the two ordered sides a path plan carries."""

    def __init__(self, sources, targets):
        self.sources, self.targets = tuple(sources), tuple(targets)


def _pairs(sources, targets):
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    return TransferAndTensorAllocator._communicating_pairs(_Plan(sources, targets))


def test_a_join_hands_every_narrow_step_to_the_same_consumer():
    """Sixteen softmax cores over eight P@V cores: consumer c is fed by c and c+8, which is
    the relation the flash bindings use to find the core holding the other half."""
    src = list(range(16))
    dst = list(range(8))
    got = {d: sorted(s for s, t in _pairs(src, dst) if t == d) for d in dst}
    assert got[0] == [0, 8]
    assert got[7] == [7, 15]


def test_a_fork_is_the_same_relation_read_the_other_way():
    src = list(range(8))
    dst = list(range(16))
    got = {s: sorted(t for a, t in _pairs(src, dst) if a == s) for s in src}
    assert got[0] == [0, 8]


def test_equal_widths_pair_one_to_one():
    assert _pairs(list(range(4)), list(range(4))) == ((0, 0), (1, 1), (2, 2), (3, 3))


def test_the_pairing_follows_the_declared_order_not_the_sorted_one():
    """Layer core lists are not normalised to ascending order, and the order is what ties a
    spatial index to a core. A rotation must pair differently, or a non-adjacent handoff
    reads as a neighbouring one."""
    rotated = _pairs(["a", "b", "c"], ["b", "c", "a"])
    assert rotated == (("a", "b"), ("b", "c"), ("c", "a"))


def test_a_core_of_another_namespace_is_not_given_aie2_adjacency(aie2):
    """An accelerator can mix namespaces, and the dispatch asks every strategy in turn."""
    assert aie2.shares_memory(_core(0, "compute", 0, 2), _core(1, "compute", 0, 3, "other")) is False
