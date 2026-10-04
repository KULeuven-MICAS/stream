from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from stream.api import SolveOptions
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import families
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE, load_families, parse_spec
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.solver import ConstraintSelection

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
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


def solve(
    solved_allocator: Callable[..., Any], two_conv: TwoConvWorkloadConfig, out: Any, options: SolveOptions
) -> Any:
    workload, mapping = make_2_conv_workload(two_conv), make_2_conv_mapping(two_conv)
    return solved_allocator(
        ACCELERATOR,
        workload,
        str(out),
        mapping,
        options,
        families_available={CapIteration.name: CapIteration},
        hook="_build_model",
    )


def test_spec_forms_parse_to_name_and_options() -> None:
    assert parse_spec("memory_ports") == ("memory_ports", {})
    assert parse_spec({"memory_ports": {"burst": False}}) == ("memory_ports", {"burst": False})
    assert parse_spec({"memory_ports": None}) == ("memory_ports", {})
    with pytest.raises(ValueError, match="exactly one family"):
        parse_spec({"a": {}, "b": {}})


@pytest.fixture
def cap_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(families, "available_families", lambda: {CapIteration.name: CapIteration})


def test_unknown_family_names_the_available_ones(cap_only: None) -> None:
    with pytest.raises(KeyError, match="available: cap_iteration"):
        load_families(["cap_iterations"])


def test_family_selected_twice_is_rejected(cap_only: None) -> None:
    with pytest.raises(ValueError, match="selected twice"):
        load_families(["cap_iteration", {"cap_iteration": {"cap": 1}}])


def test_no_families_loads_nothing() -> None:
    assert load_families([]) == ()


def test_solve_options_fold_families_into_the_constraint_selection() -> None:
    assert SolveOptions().resolved_constraint_selection() is None
    selection = SolveOptions(families=["memory_ports"]).resolved_constraint_selection()
    assert selection == ConstraintSelection(families=("memory_ports",))


@pytest.mark.slow
def test_family_adds_its_constraint_quantity_and_slot_bound(
    solved_allocator: Callable, two_conv: TwoConvWorkloadConfig, model_size: Callable, tmp_path: Any
) -> None:
    base = solve(solved_allocator, two_conv, tmp_path / "base", SolveOptions())
    options = SolveOptions(families=[{"cap_iteration": {"cap": 1e9}}])
    with_family = solve(solved_allocator, two_conv, tmp_path / "family", options)
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
        load_families("cap_iteration")


def test_solve_options_reject_a_bare_family_name() -> None:
    with pytest.raises(TypeError, match="list of names"):
        SolveOptions(families="memory_ports").resolved_constraint_selection()
