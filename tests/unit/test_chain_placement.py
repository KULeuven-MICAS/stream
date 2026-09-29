from types import SimpleNamespace

from stream.compiler.kernels.library import KernelLibrary
from stream.mapping.chain_placement import (
    bandwidth_bound,
    column_budget_options,
    is_matmul,
    layer_cost,
    row_counts,
    widest_columns,
)


def test_equal_width_beats_a_wider_softmax():
    assert row_counts([1730, 4400, 1536], 4) == (1, 1, 1)


def test_a_dominant_final_layer_widens_free_of_the_handover_tax():
    assert row_counts([100, 1000], 4) == (1, 3)


def test_a_dominant_middle_layer_does_not_widen():
    assert row_counts([1730, 4400 * 2, 1536], 4) == (1, 1, 1)


def test_a_state_consumer_is_never_wider_than_its_producer():
    costs = [1595, 4436, 11500]
    assert row_counts(costs, 4) == (1, 1, 2)
    assert row_counts(costs, 4, state_consumers=frozenset({2})) == (1, 1, 1)


def test_widest_columns_honours_granularity():
    assert widest_columns(512, 64, 1, 8) == 8
    assert widest_columns(256, 64, 1, 8) == 4
    assert widest_columns(512, 64, 2, 8) == 4


def test_column_budgets_reproduce_the_swiglu_split():
    gemm, elt = 1730.0, 128.0
    assert column_budget_options([gemm, gemm, elt, elt, gemm], 8, limit=1) == [(2, 2, 1, 1, 2)]


def _kernel(library, symbol, **shape):
    spec = library.spec(symbol)
    tile = [(position, shape[d.name], d) for position, d in enumerate(spec.dims)]
    return SimpleNamespace(
        spec=spec, library=library, call_shape=lambda: shape, call_tile=lambda: tile, operand_layouts=lambda: ()
    )


LIBRARY = KernelLibrary.from_dict(
    {
        "family": {"matmul": {"ops_per_cycle": 151.0}, "vector": {"ops_per_cycle": 16.0}},
        "kernel": {
            "gemm": {"family": "matmul", "dims": [{"name": "k"}, {"name": "n"}, {"name": "m"}]},
            "silu": {
                "family": "vector",
                "dims": [{"name": "n"}, {"name": "m"}],
                "per_op": {"cycles": 2503, "ops": 2048},
            },
        },
    }
)


def test_a_layer_is_priced_from_its_measured_call_or_its_family_rate():
    assert layer_cost(_kernel(LIBRARY, "silu", m=32, n=128)) == 2 * 2503
    assert layer_cost(_kernel(LIBRARY, "gemm", m=64, k=64, n=64)) == 64**3 / 151.0


def test_only_a_vector_kernel_can_be_bandwidth_bound():
    assert not bandwidth_bound(_kernel(LIBRARY, "gemm", m=4, k=4, n=4))
    assert is_matmul(_kernel(LIBRARY, "gemm", m=4, k=4, n=4))
    assert not is_matmul(_kernel(LIBRARY, "silu", m=32, n=64))
