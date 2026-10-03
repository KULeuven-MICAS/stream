from collections.abc import Callable
from typing import Any

import pytest

from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


@pytest.fixture(scope="module")
def alloc(
    tmp_path_factory: pytest.TempPathFactory, solved_allocator: Callable[..., Any], two_conv: TwoConvWorkloadConfig
) -> tta.TransferAndTensorAllocator:
    out = str(tmp_path_factory.mktemp("two_conv"))
    return solved_allocator(ACCELERATOR, make_2_conv_workload(two_conv), out, make_2_conv_mapping(two_conv))


def test_registry_returns_scalar_and_indexed_quantities() -> None:
    registry = QuantityRegistry()
    registry.add("overlap", 3, upper_bound=10)
    registry.add("slot_latency", 1, index=0)
    registry.add("slot_latency", 2, index=1)
    assert registry.get("overlap").upper_bound == 10
    assert registry.get("slot_latency", 1).expr == 2
    assert set(registry.indexed("slot_latency")) == {0, 1}
    assert registry.names() == ["overlap", "slot_latency"]


def test_registry_rejects_duplicates_and_names_the_available_quantities() -> None:
    registry = QuantityRegistry()
    registry.add("overlap", 3)
    with pytest.raises(ValueError, match="already registered"):
        registry.add("overlap", 4)
    with pytest.raises(KeyError, match="available: overlap"):
        registry.get("buffering")


@pytest.mark.slow
def test_allocator_registers_what_it_builds(alloc: tta.TransferAndTensorAllocator) -> None:
    expected = {"slot_latency", "overlap", "iteration", "transfer_latency", "primary"}
    expected |= {"offchip_traffic", "buffering", "route_hops"}
    assert expected <= set(alloc.quantities.names())
    slots = alloc.quantities.indexed("slot_latency")
    iteration = alloc.model.value(alloc.quantities.get("iteration").expr)
    assert iteration == sum(alloc.model.value(q.expr) for q in slots.values())


@pytest.mark.slow
def test_primary_cost_is_the_solved_primary_quantity(alloc: tta.TransferAndTensorAllocator) -> None:
    bandwidth = alloc._offchip_bandwidth()
    weight = alloc.iterations / bandwidth if bandwidth else 0.0
    traffic = alloc.model.value(alloc.quantities.get("offchip_traffic").expr)
    dma = float(alloc.max_core_dma_in.X) + float(alloc.max_core_dma_out.X)
    assert alloc.primary_cost() == pytest.approx(float(alloc.total_latency.X) + dma + weight * traffic)
