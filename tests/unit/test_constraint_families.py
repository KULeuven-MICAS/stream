from typing import Any, ClassVar

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import families
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE, load_families, parse_spec
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.solver import ConstraintSelection

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
PRESSURE_BOUND = 10**9


class CapIteration:
    """Test family: registers the iteration as a quantity and caps it at ``cap`` cycles."""

    name: ClassVar[str] = "cap_iteration"

    def __init__(self, cap: float = 1e12) -> None:
        self.cap = cap

    def declare(self, alloc: tta.TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        q.add(SLOT_PRESSURE, 0, index="test", upper_bound=PRESSURE_BOUND)

    def constrain(self, alloc: tta.TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        iteration = q.get("iteration").expr
        q.add("capped_iteration", iteration)
        alloc.model.add_constr(iteration <= self.cap, name="cap_iteration")


def solve(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, options: SolveOptions) -> tta.TransferAndTensorAllocator:
    monkeypatch.setattr(families, "available_families", lambda: {CapIteration.name: CapIteration})
    built: list[tta.TransferAndTensorAllocator] = []
    original = tta.TransferAndTensorAllocator._build_model

    def capture(self: tta.TransferAndTensorAllocator) -> None:
        built.append(self)
        original(self)

    monkeypatch.setattr(tta.TransferAndTensorAllocator, "_build_model", capture)
    workload, mapping = make_2_conv_workload(TWO_CONV), make_2_conv_mapping(TWO_CONV)
    evaluate_mapping(ACCELERATOR, workload, str(tmp_path), mapping, options)
    return built[0]


def model_size(alloc: tta.TransferAndTensorAllocator) -> tuple[int, int]:
    raw = alloc.model._model  # type: ignore[attr-defined]
    return sum(1 for _ in raw.variables()), sum(1 for _ in raw.linear_constraints())


def test_spec_forms_parse_to_name_and_options() -> None:
    assert parse_spec("energy") == ("energy", {})
    assert parse_spec({"memory_ports": {"burst": False}}) == ("memory_ports", {"burst": False})
    assert parse_spec({"memory_ports": None}) == ("memory_ports", {})
    with pytest.raises(ValueError, match="exactly one family"):
        parse_spec({"a": {}, "b": {}})


def test_unknown_family_names_the_available_ones() -> None:
    with pytest.raises(KeyError, match="available: cap_iteration"):
        load_families(["cap_iterations"], {CapIteration.name: CapIteration})


def test_family_selected_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="selected twice"):
        load_families(["cap_iteration", {"cap_iteration": {"cap": 1}}], {CapIteration.name: CapIteration})


def test_no_families_loads_nothing() -> None:
    assert load_families([], {}) == ()


def test_solve_options_fold_families_into_the_constraint_selection() -> None:
    assert SolveOptions().resolved_constraint_selection() is None
    selection = SolveOptions(families=["energy"]).resolved_constraint_selection()
    assert selection == ConstraintSelection(families=("energy",))


@pytest.mark.slow
def test_family_adds_its_constraint_quantity_and_slot_bound(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    base = solve(monkeypatch, tmp_path / "base", SolveOptions())
    with_family = solve(monkeypatch, tmp_path / "family", SolveOptions(families=[{"cap_iteration": {"cap": 1e9}}]))
    base_vars, base_cons = model_size(base)
    family_vars, family_cons = model_size(with_family)
    assert family_vars == base_vars
    assert family_cons == base_cons + 1
    assert "capped_iteration" in with_family.quantities
    assert "capped_iteration" not in base.quantities
    assert with_family._family_slot_pressure_bound() == PRESSURE_BOUND
    assert base._family_slot_pressure_bound() == 0


def test_a_bare_family_name_is_rejected() -> None:
    with pytest.raises(TypeError, match="list of names"):
        load_families("cap_iteration", {CapIteration.name: CapIteration})


def test_solve_options_reject_a_bare_family_name() -> None:
    with pytest.raises(TypeError, match="list of names"):
        SolveOptions(families="memory_ports").resolved_constraint_selection()
