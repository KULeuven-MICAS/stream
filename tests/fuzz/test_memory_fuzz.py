"""A few seeds of the memory fuzzer; ``python -m tests.fuzz.memory_fuzz`` runs more."""

import pytest

from tests.fuzz.memory_fuzz import run


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
def test_a_random_accelerator_solves_soundly(seed):
    result = run(seed)
    assert result["status"] in ("ok", "route_sensitive", "infeasible"), result


@pytest.mark.slow
def test_the_tiling_leaves_room_for_what_the_allocation_pins():
    """Seed 977, a MoE on small memories, solves with room for the tensors the allocation keeps resident."""
    assert run(977)["status"] == "ok"
