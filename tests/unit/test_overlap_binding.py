"""Tests for the solver's overlap evidence (``_overlap_section``).

``binding_resources`` is the answer to "what limits the pipelining", and it is only useful if it
is never spuriously empty: a consumer reading an empty list concludes nothing binds the overlap,
which is never true of a solved schedule.
"""

from __future__ import annotations

from types import SimpleNamespace

from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    TransferAndTensorAllocator,
)


def make_allocator(slack: dict[object, int], overlap: int | None, recurrence: int = 0):
    """An allocator stub carrying only what the overlap section reads, so no solve is needed."""
    allocator = object.__new__(TransferAndTensorAllocator)
    allocator.idle_lat = {res: SimpleNamespace(X=float(cycles)) for res, cycles in slack.items()}
    allocator.overlap = None if overlap is None else SimpleNamespace(X=float(overlap))
    allocator.recurrence_bound = recurrence
    return allocator


def core(core_id: int) -> Core:
    return Core(core_id=core_id, name=f"core_{core_id}", core_type="zigzag.compute")


class TestOverlapSection:
    def test_binding_is_the_argmin_when_overlap_sits_below_the_cap(self) -> None:
        """The real TPU7x SwiGLU numbers: the overlap solves to 14923 while the tightest resource has
        14942 cycles of slack. The primary objective is a sum in which latency trades against DMA
        balance, so the optimum need not push the overlap onto its bound -- an equality test reports
        an empty binding set on a perfectly ordinary run."""
        allocator = make_allocator({core(0): 14942, core(1): 181228, core(2): 14942}, overlap=14923)

        section = allocator._overlap_section()

        assert section["overlap_cycles"] == 14923
        assert section["binding_resources"] == [str(core(0)), str(core(2))]

    def test_binding_matches_an_equality_test_when_the_overlap_is_at_its_cap(self) -> None:
        """Where the old rule worked it must keep working: at the cap, argmin and equality agree."""
        allocator = make_allocator({core(0): 100, core(1): 250}, overlap=100)

        section = allocator._overlap_section()

        assert section["binding_resources"] == [str(core(0))]

    def test_zero_slack_pins_the_overlap_to_zero(self) -> None:
        """A resource busy from an early to a late slot has no boundary idle and binds at 0."""
        allocator = make_allocator({core(0): 0, core(1): 5000}, overlap=0)

        section = allocator._overlap_section()

        assert section["overlap_cycles"] == 0
        assert section["binding_resources"] == [str(core(0))]

    def test_no_resources_yields_no_binding_set(self) -> None:
        """Nothing to be binding is the one case where empty is the honest answer."""
        allocator = make_allocator({}, overlap=None)

        section = allocator._overlap_section()

        assert section["binding_resources"] == []
        assert section["per_resource_slack"] == []

    def test_recurrence_bound_is_carried_through(self) -> None:
        allocator = make_allocator({core(0): 100}, overlap=64, recurrence=64)

        assert allocator._overlap_section()["recurrence_bound_cycles"] == 64
