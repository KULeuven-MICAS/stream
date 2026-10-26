"""A few seeds of the memory fuzzer (``python -m tests.fuzz.memory_fuzz`` runs as many as asked): each must solve or
be infeasible with a reason, keep what placing tensors in memories promises, and not depend on the order the
description lists its memories in."""

import pytest

from tests.fuzz.memory_fuzz import run


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
def test_a_random_accelerator_solves_soundly(seed):
    result = run(seed)
    assert result["status"] in ("ok", "route_sensitive", "infeasible"), result


@pytest.mark.slow
def test_the_tiling_leaves_room_for_what_the_allocation_pins():
    """Seed 977: a MoE on small memories, whose expert outputs and dispatched tokens the allocation keeps resident
    across their outermost loops; the tiling must leave room for them, rather than count two buffers each."""
    assert run(977)["status"] == "ok"
