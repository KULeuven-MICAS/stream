from typing import Any

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import families
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.families.memory_ports import MemoryPorts

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
