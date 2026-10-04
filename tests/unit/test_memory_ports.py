from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.cost_model.port_activity import PortActivity
from zigzag.datatypes import LayerOperand
from zigzag.hardware.architecture.memory_port import DataDirection

from stream.api import SolveOptions
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES, traffic
from stream.opt.allocation.constraint_optimization.families.memory_ports import MemoryPorts
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.workload.steady_state.iteration_space import LoopEffect

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
DRAM = (6, "dram", "rw_port_1")
Solve = Callable[[list[Any]], tta.TransferAndTensorAllocator]


@pytest.fixture(scope="module")
def solve(
    tmp_path_factory: pytest.TempPathFactory, solved_allocator: Callable[..., Any], two_conv: TwoConvWorkloadConfig
) -> Solve:
    def run(family_specs: list[Any]) -> tta.TransferAndTensorAllocator:
        return solved_allocator(
            ACCELERATOR,
            make_2_conv_workload(two_conv),
            str(tmp_path_factory.mktemp("two_conv")),
            make_2_conv_mapping(two_conv),
            SolveOptions(families=[*DEFAULT_FAMILIES, *family_specs]),
        )

    return run


@pytest.fixture(scope="module")
def base(solve: Solve) -> tta.TransferAndTensorAllocator:
    return solve([])


@pytest.fixture(scope="module")
def ports(solve: Solve) -> tta.TransferAndTensorAllocator:
    return solve(["memory_ports"])


def rate(alloc: tta.TransferAndTensorAllocator, key: Any) -> float:
    return alloc.quantities.get("port_rate", key).expr


@pytest.mark.slow
def test_memory_ports_add_constraints_but_no_variables(base: Any, ports: Any, model_size: Callable) -> None:
    (base_vars, base_cons), (port_vars, port_cons) = model_size(base), model_size(ports)
    assert port_vars == base_vars
    assert port_cons > base_cons


@pytest.mark.slow
def test_every_port_moves_its_demand_within_the_interval(ports: Any, interval: Callable) -> None:
    for key, demand in ports.quantities.indexed("port_demand").items():
        assert ports.model.value(demand.expr) <= rate(ports, key) * interval(ports) + 1e-6


@pytest.mark.slow
def test_every_port_moves_each_slot_demand_within_that_slot(ports: Any) -> None:
    for (key, slot), demand in ports.quantities.indexed("port_demand_slot").items():
        slot_latency = ports.model.value(ports.quantities.get("slot_latency", slot).expr)
        assert ports.model.value(demand.expr) <= rate(ports, key) * slot_latency + 1e-6


@pytest.mark.slow
def test_offchip_port_binds_and_stretches_the_schedule(base: Any, ports: Any, interval: Callable) -> None:
    assert ports.total_latency.X > base.total_latency.X
    utilisation = {
        key: ports.model.value(d.expr) / (rate(ports, key) * interval(ports))
        for key, d in ports.quantities.indexed("port_demand").items()
    }
    assert max(utilisation, key=lambda key: utilisation[key]) == DRAM


@pytest.mark.slow
def test_no_slot_outgrows_the_slot_pressure_a_port_declares(ports: Any) -> None:
    worst = max(q.upper_bound for q in ports.quantities.indexed("slot_pressure").values())
    assert all(ports.model.value(q.expr) <= worst for q in ports.quantities.indexed("slot_latency").values())


@pytest.mark.slow
def test_burst_off_builds_one_constraint_per_port(solve: Solve, base: Any, model_size: Callable) -> None:
    interval_only = solve([{"memory_ports": {"burst": False}}])
    n_ports = len(interval_only.quantities.indexed("port_demand"))
    assert model_size(interval_only)[1] == model_size(base)[1] + n_ports


@pytest.mark.slow
def test_without_bounds_the_family_only_reports(solve: Solve, base: Any, model_size: Callable) -> None:
    report_only = solve([{"memory_ports": {"interval": False, "burst": False}}])
    assert model_size(report_only) == model_size(base)
    assert report_only.total_latency.X == base.total_latency.X
    rows = report_only.compute_performance_stats()["memory_ports"]
    assert {row["kind"] for row in rows} == {"memory_port", "link"}
    assert rows[0]["utilization"] == max(row["utilization"] for row in rows)


def test_an_accelerator_without_port_models_gets_no_port_constraint() -> None:
    alloc = MagicMock()
    registry = QuantityRegistry()
    MemoryPorts().build(alloc, registry)
    alloc.model.add_constr.assert_not_called()


def port_registry() -> tuple[MagicMock, QuantityRegistry]:
    alloc = MagicMock()
    registry = QuantityRegistry()
    for name, value in (("iteration", 10), ("overlap", 2)):
        registry.add(name, value)
    registry.add("port_rate", 64.0, index=DRAM)
    registry.add("port_demand", 256.0, index=DRAM)
    registry.add("port_demand_slot", 256.0, index=(DRAM, 0))
    registry.add("slot_latency", 4, index=0)
    return alloc, registry


def constraint_names(alloc: MagicMock) -> list[str]:
    return [call.kwargs["name"] for call in alloc.model.add_constr.call_args_list]


def test_burst_off_keeps_only_the_interval_bound() -> None:
    alloc, registry = port_registry()
    MemoryPorts(burst=False).build(alloc, registry)
    assert constraint_names(alloc) == ["port_interval_6_dram_rw_port_1"]


def test_interval_off_keeps_only_the_burst_bound() -> None:
    alloc, registry = port_registry()
    MemoryPorts(interval=False).build(alloc, registry)
    assert constraint_names(alloc) == ["port_burst_6_dram_rw_port_1_0"]


def test_a_stream_whose_active_latency_rounds_to_zero_keeps_its_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    tr, choice, alloc = MagicMock(), MagicMock(), MagicMock()
    tr.inputs[0].size_bits.return_value = 4096
    alloc.quantities = QuantityRegistry()
    alloc.quantities.add("transfer_latency", "gated", index=(tr, choice))
    alloc.transfer_latency_for_path.return_value = 1
    loops = [SimpleNamespace(size=2, effect=LoopEffect.INVARIANT), SimpleNamespace(size=8, effect=LoopEffect.ABSENT)]
    alloc.ssis.get.return_value.get_temporal_variables.return_value = loops
    alloc.y_path_choice = {(tr, choice): SimpleNamespace(_raw="y")}
    alloc.slot_of = {tr: 0}
    monkeypatch.setattr(traffic, "_sides", lambda *_: [])
    (stream,) = traffic.dma_streams(alloc)
    # Active for 1 of 8 iterations: 1 cycle rounds to 0, while 4096 / 8 bits still move.
    assert (stream.active_cycles, stream.bits_per_cycle * stream.latency_ub) == ("y", 512)


def zigzag_stall(real_cycles: list[int], window: int, periods: int) -> float:
    """ZigZag's combined stall of double-buffered streams sharing one port (CostModelEvaluation, step 2)."""
    activities = [
        PortActivity(r, window, window, periods, LayerOperand("I"), 0, DataDirection.WR_IN_BY_HIGH) for r in real_cycles
    ]
    union = CostModelEvaluation._CostModelEvaluation__calc_mem_updating_window_union(None, activities)  # type: ignore[attr-defined]
    positive = sum(a.stall_or_slack for a in activities if a.stall_or_slack > 0)
    negative = sum(a.stall_or_slack for a in activities if a.stall_or_slack <= 0)
    return positive + max(0, negative + sum(a.mem_updating_window for a in activities) - union)


@pytest.mark.parametrize("bits", [[4096], [4096, 2048], [1024, 1024, 3072, 512]])
def test_interval_bound_is_zigzag_stall_free_window_of_double_buffered_streams(bits: list[int]) -> None:
    rate, periods = 64, 10_000
    real = [b // rate for b in bits]
    interval = sum(bits) / rate  # the interval bound at equality: rate * interval = the streams' bits
    assert zigzag_stall(real, round(interval), periods) == 0
    assert zigzag_stall(real, round(interval) - 1, periods) > 0
