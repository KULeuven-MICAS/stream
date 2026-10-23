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
