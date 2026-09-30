from typing import Any
from unittest.mock import MagicMock

import pytest
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.cost_model.port_activity import PortActivity
from zigzag.datatypes import LayerOperand
from zigzag.hardware.architecture.memory_port import DataDirection

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import families
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.families.memory_ports import MemoryPorts
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
TWO_CONV = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)
DRAM = (6, "dram", "rw_port_1")


def solve(tmp_path_factory: pytest.TempPathFactory, family_specs: list[Any]) -> tta.TransferAndTensorAllocator:
    solved: list[tta.TransferAndTensorAllocator] = []
    original = tta.TransferAndTensorAllocator.solve

    def capture(self: tta.TransferAndTensorAllocator, **kwargs: bool) -> object:
        solved.append(self)
        return original(self, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(families, "available_families", lambda: {MemoryPorts.name: MemoryPorts})
        patch.setattr(tta.TransferAndTensorAllocator, "solve", capture)
        out = str(tmp_path_factory.mktemp("two_conv"))
        workload, mapping = make_2_conv_workload(TWO_CONV), make_2_conv_mapping(TWO_CONV)
        evaluate_mapping(ACCELERATOR, workload, out, mapping, SolveOptions(families=family_specs))
    return solved[0]


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tta.TransferAndTensorAllocator:
    return solve(tmp_path_factory, [])


@pytest.fixture(scope="module")
def ports(tmp_path_factory: pytest.TempPathFactory) -> tta.TransferAndTensorAllocator:
    return solve(tmp_path_factory, ["memory_ports"])


def model_size(alloc: tta.TransferAndTensorAllocator) -> tuple[int, int]:
    raw = alloc.model._model  # type: ignore[attr-defined]
    return sum(1 for _ in raw.variables()), sum(1 for _ in raw.linear_constraints())


def interval(alloc: tta.TransferAndTensorAllocator) -> float:
    return alloc.model.value(alloc.quantities.get("iteration").expr) - alloc.overlap.X


@pytest.mark.slow
def test_memory_ports_add_constraints_but_no_variables(base: Any, ports: Any) -> None:
    (base_vars, base_cons), (port_vars, port_cons) = model_size(base), model_size(ports)
    assert port_vars == base_vars
    assert port_cons > base_cons


@pytest.mark.slow
def test_every_port_moves_its_demand_within_the_interval(ports: Any) -> None:
    family = ports.families[0]
    for key, demand in ports.quantities.indexed("port_demand").items():
        assert ports.model.value(demand.expr) <= family.rate[key] * interval(ports) + 1e-6


@pytest.mark.slow
def test_every_port_moves_each_slot_demand_within_that_slot(ports: Any) -> None:
    family = ports.families[0]
    for (key, slot), demand in ports.quantities.indexed("port_demand_slot").items():
        assert ports.model.value(demand.expr) <= family.rate[key] * ports.slot_latency[slot].X + 1e-6


@pytest.mark.slow
def test_offchip_port_binds_and_stretches_the_schedule(base: Any, ports: Any) -> None:
    # Prototype evidence (evidence_59a9340/ports.jsonl, test_co_tpu_two_conv): 12808 -> 18072 cycles, DRAM binding.
    assert base.total_latency.X == 12808
    assert ports.total_latency.X == 18072
    utilisation = {
        key: ports.model.value(d.expr) / (ports.families[0].rate[key] * interval(ports))
        for key, d in ports.quantities.indexed("port_demand").items()
    }
    assert max(utilisation, key=lambda key: utilisation[key]) == DRAM


@pytest.mark.slow
def test_slot_big_m_covers_the_longest_slot_a_port_can_force(ports: Any) -> None:
    worst_demand_ub = max(q.upper_bound or 0.0 for q in ports.quantities.indexed("slot_pressure").values())
    assert ports._family_slot_pressure_bound() == worst_demand_ub
    assert all(v.X <= worst_demand_ub for v in ports.slot_latency.values())


@pytest.mark.slow
def test_burst_off_builds_one_constraint_per_port(tmp_path_factory: pytest.TempPathFactory, base: Any) -> None:
    interval_only = solve(tmp_path_factory, [{"memory_ports": {"burst": False}}])
    n_ports = len(interval_only.quantities.indexed("port_demand"))
    assert model_size(interval_only)[1] == model_size(base)[1] + n_ports


def test_an_accelerator_without_port_models_gets_no_port_constraint() -> None:
    alloc = MagicMock()
    registry = QuantityRegistry()
    MemoryPorts().constrain(alloc, registry)
    alloc.model.add_constr.assert_not_called()


def port_registry(transfer_contention: bool) -> tuple[MagicMock, QuantityRegistry]:
    alloc = MagicMock()
    alloc.constraint_selection.transfer_contention = transfer_contention
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


@pytest.mark.parametrize("transfer_contention", [True, False])
def test_the_interval_bound_holds_with_or_without_transfer_contention(transfer_contention: bool) -> None:
    alloc, registry = port_registry(transfer_contention)
    MemoryPorts(burst=False).constrain(alloc, registry)
    assert constraint_names(alloc) == ["port_interval_6_dram_rw_port_1"]


def test_interval_off_keeps_only_the_burst_bound() -> None:
    alloc, registry = port_registry(transfer_contention=True)
    MemoryPorts(interval=False).constrain(alloc, registry)
    assert constraint_names(alloc) == ["port_burst_6_dram_rw_port_1_0"]


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
    interval = sum(bits) / rate  # P2 at equality: rate * interval = sum of the streams' bits
    assert zigzag_stall(real, round(interval), periods) == 0
    assert zigzag_stall(real, round(interval) - 1, periods) > 0
