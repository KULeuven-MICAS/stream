from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from stream.api import SolveOptions, default_families
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.opt.allocation.constraint_optimization import families
from stream.opt.allocation.constraint_optimization import transfer_and_tensor_allocation as tta
from stream.opt.allocation.constraint_optimization.context import AIE2Constraints
from stream.opt.allocation.constraint_optimization.families import (
    DEFAULT_FAMILIES,
    SLOT_PRESSURE,
    drop_families,
    load_families,
    parse_spec,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
AIE = "stream/inputs/aie/hardware/whole_array_strix.yaml"
PRESSURE_BOUND = 10**9


class CapIteration:
    """Test family: declares a slot pressure before the overlap, and caps the iteration at ``cap`` cycles after."""

    name: ClassVar[str] = "cap_iteration"
    declare_requires: ClassVar[tuple[str, ...]] = ()
    declares: ClassVar[tuple[str, ...]] = (SLOT_PRESSURE,)
    requires: ClassVar[tuple[str, ...]] = ("iteration",)
    provides: ClassVar[tuple[str, ...]] = ("capped_iteration",)

    def __init__(self, cap: float = 1e12) -> None:
        self.cap = cap

    def declare(self, alloc: tta.TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        q.add(SLOT_PRESSURE, 0, index="test", upper_bound=PRESSURE_BOUND)

    def build(self, alloc: tta.TransferAndTensorAllocator, q: QuantityRegistry) -> None:
        iteration = q.get("iteration").expr
        q.add("capped_iteration", iteration)
        alloc.model.add_constr(iteration <= self.cap, name="cap_iteration")


class Needs:
    """Test family that requires one name and provides another."""

    def __init__(self, name: str, requires: tuple[str, ...], provides: tuple[str, ...] = ()) -> None:
        self.name, self.requires, self.provides = name, requires, provides

    def build(self, alloc: Any, q: QuantityRegistry) -> None: ...


@pytest.fixture
def with_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    known = {**families.available_families(), CapIteration.name: CapIteration}
    monkeypatch.setattr(families, "available_families", lambda: known)


def order(specs: Any) -> list[str]:
    return [name for name, _ in load_families(specs).steps]


def test_families_are_discovered_once_per_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep resolves the families of every solve, so the entry points are scanned once."""
    calls: list[Any] = []
    monkeypatch.setattr(families, "load_group", lambda group, allow: calls.append(allow) or [])
    families._discovered.cache_clear()
    try:
        families.available_families()
        families.available_families()
        assert len(calls) == 1
    finally:
        families._discovered.cache_clear()


def test_spec_forms_parse_to_name_and_options() -> None:
    assert parse_spec("memory_ports") == ("memory_ports", {})
    assert parse_spec({"memory_ports": {"burst": False}}) == ("memory_ports", {"burst": False})
    assert parse_spec({"memory_ports": None}) == ("memory_ports", {})
    with pytest.raises(ValueError, match="exactly one family"):
        parse_spec({"a": {}, "b": {}})


def test_unknown_family_names_the_available_ones() -> None:
    with pytest.raises(KeyError, match="available: .*memory_ports"):
        load_families(["memory_port"])


def test_family_selected_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="selected twice"):
        load_families([*DEFAULT_FAMILIES, {"overlap": {"model": "span"}}])


def test_no_families_loads_nothing() -> None:
    assert load_families([]).families == ()


def test_a_bare_family_name_is_rejected() -> None:
    with pytest.raises(TypeError, match="list of names"):
        load_families("memory_ports")


def test_an_unknown_option_names_its_family() -> None:
    with pytest.raises(TypeError, match="'overlap'"):
        load_families([*(f for f in DEFAULT_FAMILIES if f != "overlap"), {"overlap": {"modle": "span"}}])


def test_the_defaults_are_streams_families_and_the_namespaces() -> None:
    assert SolveOptions().families is None
    assert default_families(ACCELERATOR) == DEFAULT_FAMILIES
    assert default_families(AIE) == (*DEFAULT_FAMILIES, *AIE2Constraints.families)


def test_streams_families_build_in_their_declared_order() -> None:
    assert order(DEFAULT_FAMILIES) == list(DEFAULT_FAMILIES)
    assert order(DEFAULT_FAMILIES[::-1]) == list(DEFAULT_FAMILIES)


def test_a_namespace_family_runs_right_after_what_it_bounds() -> None:
    steps = order(default_families(AIE)[::-1])
    for bound, family in (
        ("object_fifo_depth", "aie2_object_fifo_depth"),
        ("buffer_descriptors", "aie2_buffer_descriptors"),
        ("reuse_compatibility", "aie2_memory_reuse"),
        ("dma_channels", "aie2_dma_channels"),
    ):
        assert steps.index(family) == steps.index(bound) + 1


def test_a_declaring_family_runs_its_declare_before_the_overlap_and_its_build_after() -> None:
    steps = order([*DEFAULT_FAMILIES, "memory_ports"])
    declare, build = (i for i, name in enumerate(steps) if name == "memory_ports")
    assert steps.index("slot_latency") < declare < steps.index("overlap") < build < steps.index("dma_channels")


def test_a_requirement_no_family_provides_is_rejected() -> None:
    with pytest.raises(ValueError, match="'slot_latency' requires 'reuse_factor'"):
        load_families([f for f in DEFAULT_FAMILIES if f != "reuse_rates"])


def test_families_that_require_each_other_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    known = {"a": lambda: Needs("a", ("y",), ("x",)), "b": lambda: Needs("b", ("x",), ("y",))}
    monkeypatch.setattr(families, "available_families", lambda: known)
    with pytest.raises(ValueError, match=r"\['a', 'b'\] require each other"):
        load_families(["a", "b"])


def test_dropping_a_family_drops_those_that_need_it() -> None:
    kept = drop_families(default_families(AIE), ["object_fifo_depth", "dma_channels"])
    assert {"object_fifo_depth", "aie2_object_fifo_depth", "dma_channels", "aie2_dma_channels"}.isdisjoint(kept)
    assert "aie2_buffer_descriptors" in kept
    assert default_families(AIE, without=["overlap"]) == tuple(f for f in default_families(AIE) if f != "overlap")
    assert "memory_ports" not in drop_families([*DEFAULT_FAMILIES, "memory_ports"], ["overlap"])


@pytest.mark.slow
def test_family_adds_its_constraint_quantity_and_slot_bound(
    solved_allocator: Callable, two_conv: TwoConvWorkloadConfig, model_size: Callable, tmp_path: Any, with_cap: None
) -> None:
    workload, mapping = make_2_conv_workload(two_conv), make_2_conv_mapping(two_conv)

    def solve(path: str, options: SolveOptions) -> Any:
        return solved_allocator(ACCELERATOR, workload, path, mapping, options, hook="_build_model")

    base = solve(str(tmp_path / "base"), SolveOptions())
    options = SolveOptions(families=[*DEFAULT_FAMILIES, {"cap_iteration": {"cap": 1e9}}])
    with_family = solve(str(tmp_path / "family"), options)
    base_vars, base_cons = model_size(base)
    family_vars, family_cons = model_size(with_family)
    assert family_vars == base_vars
    assert family_cons == base_cons + 1
    assert "capped_iteration" in with_family.quantities
    assert "capped_iteration" not in base.quantities
    assert with_family._slot_pressure_bound() == PRESSURE_BOUND
    assert base._slot_pressure_bound() < PRESSURE_BOUND
