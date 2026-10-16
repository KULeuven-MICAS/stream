from types import SimpleNamespace

import pytest

from stream.compiler.kernels.library import KernelLibrary
from stream.stages.estimation.aie_cost_estimator import AIECostEstimator

LIBRARY = KernelLibrary.from_dict(
    {
        "family": {
            "matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}},
            "vector": {"ops_per_cycle": 16.0},
        },
        "kernel": {
            "matmul_bf16_bf16": {"family": "matmul", "dims": [{"name": "k"}, {"name": "n"}, {"name": "m"}]},
            "silu_bf16": {"family": "vector", "dims": [{"name": "m"}]},
        },
    }
)


@pytest.mark.parametrize(
    ("kernel", "ops"),
    [
        (SimpleNamespace(library=LIBRARY, function_name="matmul_bf16_bf16"), 512),
        (SimpleNamespace(library=LIBRARY, function_name="silu_bf16"), None),
        (SimpleNamespace(library=None, function_name="matmul_bf16_bf16"), None),
        (None, None),
    ],
)
def test_a_matmul_kernel_is_ideal_at_one_mac_tile_per_cycle(kernel, ops):
    """An 8x8x8 MAC tile is 512 MACs a cycle, the peak a measured call of the kernel is judged against; a vector
    kernel, or a node without a library, keeps the per-datatype rate."""
    assert AIECostEstimator._mac_unit_ops(kernel) == ops
