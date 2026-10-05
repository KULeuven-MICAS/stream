"""The allocation model's own part: the families a selection builds, the objective levels they contribute, the
overlap formulation and the buffering helpers."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stream.api import SolveOptions, default_families
from stream.hardware.architecture.core import Core
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.opt.allocation.constraint_optimization.families import (
    DEFAULT_FAMILIES,
    LATENCY,
    available_families,
    drop_families,
    load_families,
    overlap,
)
from stream.opt.allocation.constraint_optimization.families.overlap import PipeliningModel
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import ObjectiveLevel
from stream.workload.steady_state.iteration_space import Reuse

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
Solve = Callable[..., AllocationModel]


@pytest.fixture(scope="module")
def solve(tmp_path_factory: pytest.TempPathFactory, solved_model: Solve, two_conv: TwoConvWorkloadConfig) -> Solve:
    """A solve of the two-convolution workload on the TPU-like array with ``families``."""
    workload, mapping = make_2_conv_workload(two_conv), make_2_conv_mapping(two_conv)

    def run(families: Any, **kwargs: Any) -> AllocationModel:
        out = str(tmp_path_factory.mktemp("objective"))
        options = SolveOptions(families=families, artifacts=False)
        return solved_model(ACCELERATOR, workload, out, mapping, options, **kwargs)

    return run


@pytest.fixture(scope="module")
def models(solve: Solve) -> dict[str, AllocationModel]:
    return {"default": solve(None), "no_dma": solve(default_families(ACCELERATOR, ["dma_channels"]))}


def _value(model: AllocationModel, name: str) -> float:
    return model.model.value(model.quantities.get(name).expr) if name in model.quantities else 0.0


def _latency_level(model: AllocationModel) -> tuple[float, float]:
    """The solved latency level, and what it is made of: the latency, the DMA peaks and the off-chip charge."""
    value, objective = model.model.value, model.objective
    charge = _value(model, "offchip_traffic_weight") * value(objective["offchip_traffic"].expr)
    parts = _value(model, "total_latency") + _value(model, "dma_peak_in") + _value(model, "dma_peak_out") + charge
    return value(objective["latency"].expr), parts


@pytest.mark.parametrize("case", ["default", "no_dma"])
def test_the_objective_levels_come_in_priority_order(models: dict[str, AllocationModel], case: str):
    """Latency decides first; offchip traffic breaks its ties, buffering breaks traffic's, and the route length
    breaks buffering's."""
    objective = models[case].objective
    assert list(objective) == ["latency", "offchip_traffic", "buffering", "route_hops"]
    assert [level.priority for level in objective.values()] == [4, 3, 2, 1]


def test_the_latency_level_sums_its_families_contributions(models: dict[str, AllocationModel]):
    """The run's latency, the DMA peaks where dma_channels is selected, and the weighted off-chip traffic."""
    for model in models.values():
        solved, parts = _latency_level(model)
        assert solved == pytest.approx(parts)
    assert "dma_peak_in" in models["default"].quantities
    assert "dma_peak_in" not in models["no_dma"].quantities


def test_a_family_left_out_builds_nothing():
    """Leaving a family out of the selection is what switches its constraints and objective terms off."""
    selection = load_families(drop_families(DEFAULT_FAMILIES, ["memory_capacity", "dma_channels"]))
    assert {"memory_capacity", "dma_channels"}.isdisjoint(name for name, _ in selection.steps)
    assert {"memory_capacity", "dma_channels"}.isdisjoint(family.name for family in selection.families)


def test_the_capacity_screen_runs_whatever_the_families(solve: Solve):
    """A memory too small for its pinned tensors fails the solve before the model is built, with memory_capacity
    left out too, as on 1.x."""
    with (
        patch.object(Core, "get_memory_capacity", return_value=1),
        pytest.raises(InfeasibleAllocationError, match="pinned to it"),
    ):
        solve(default_families(ACCELERATOR, ["memory_capacity"]), hook="_build_model")


def _offchip_traffic(**options: Any) -> Any:
    return available_families()["offchip_traffic"](**options)


def test_offchip_traffic_is_charged_in_the_primary_objective():
    """With the family's weight registered, the bytes join the latency in the primary objective."""
    ctx = SimpleNamespace(
        model=MagicMock(quicksum=MagicMock(return_value=SimpleNamespace(_raw=2048))),
        space=SimpleNamespace(tensors_to_optimize_reuse_for=[]),
        vars=SimpleNamespace(z_stop={}),
        quantities=QuantityRegistry(),
    )
    family = _offchip_traffic()
    assert [level.name for level in family.objective(ctx)] == ["offchip_traffic"]
    ctx.quantities.add("offchip_traffic_weight", 1 / 512)
    traffic, latency = family.objective(ctx)
    assert (traffic.expr, latency.name, latency.priority, latency.expr) == (2048, "latency", LATENCY, 4)


@pytest.mark.parametrize(
    ("shared_bandwidth", "bandwidth", "charged"),
    [({}, 512.0, True), ({}, 0.0, False), ({0: object()}, 512.0, False)],
    ids=["offchip_links", "no_offchip_core", "shared_bandwidth_model"],
)
def test_offchip_traffic_weight(shared_bandwidth, bandwidth, charged):
    """No off-chip core charges nothing, and with a shared-bandwidth model the bytes are in the latency already."""
    space = SimpleNamespace(shared_bandwidth=shared_bandwidth, iterations=4, offchip_bandwidth=lambda: bandwidth)
    q = QuantityRegistry()
    _offchip_traffic().build(SimpleNamespace(space=space, quantities=q))
    assert ("offchip_traffic_weight" in q) is charged
    if charged:
        assert q.get("offchip_traffic_weight").expr == 4 / bandwidth


def test_offchip_traffic_without_its_charge_keeps_its_level():
    """``charge`` False drops only the latency charge, as ConstraintSelection(offchip_traffic_cost=False) did."""
    space = SimpleNamespace(shared_bandwidth={}, iterations=4, offchip_bandwidth=lambda: 512.0)
    q = QuantityRegistry()
    _offchip_traffic(charge=False).build(SimpleNamespace(space=space, quantities=q))
    assert "offchip_traffic_weight" not in q


def test_a_selection_without_the_overlap_is_rejected_before_any_stage_runs():
    with pytest.raises(ValueError, match="define no total_latency; add overlap"):
        load_families(drop_families(DEFAULT_FAMILIES, ["overlap", "dma_channels"]))


class _Rogue:
    name = "rogue"
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def build(self, ctx: Any) -> None: ...

    def objective(self, ctx: Any) -> list[ObjectiveLevel]:
        return [ObjectiveLevel(expr=1, priority=LATENCY + 1, name="latency")]


def test_levels_of_one_name_must_share_a_priority(solve: Solve):
    families = {**available_families(), "rogue": _Rogue}
    with pytest.raises(ValueError, match="'latency' has priority 5 in 'rogue'"):
        solve([*DEFAULT_FAMILIES, "rogue"], families_available=families, hook="_build_model")


def _overlap(spec) -> PipeliningModel:
    selection = load_families(["reuse_rates", "slot_latency", spec])
    return next(f for f in selection.families if f.name == "overlap").model


def test_pipelining_defaults_to_occupancy():
    """The modulo-scheduling model is the default; span is the opt-in legacy one."""
    assert _overlap("overlap") is PipeliningModel.OCCUPANCY
    assert _overlap({"overlap": {"model": "span"}}) is PipeliningModel.SPAN


@pytest.mark.parametrize(
    ("model", "expected_builder", "other_builder"),
    [
        (PipeliningModel.OCCUPANCY, "_occupancy_indicators", "_span_indicators"),
        (PipeliningModel.SPAN, "_span_indicators", "_occupancy_indicators"),
    ],
)
def test_idle_indicator_dispatch(model, expected_builder, other_builder, monkeypatch: pytest.MonkeyPatch):
    """Each model builds its own indicators and only its own."""
    builders = {name: MagicMock() for name in (expected_builder, other_builder)}
    for name, builder in builders.items():
        monkeypatch.setattr(overlap, name, builder)
    monkeypatch.setattr(overlap, "_resource_activity", lambda ctx: [("res", [0], "used")])
    overlap._idle_indicators(MagicMock(), model)
    builders[expected_builder].assert_called_once()
    builders[other_builder].assert_not_called()


class _FakeTensor:
    """Hashable stand-in -- the helper keys its dicts by tensor."""

    name = "t"


def _fire_helper_stub(*, relevant_sizes):
    """A decision space carrying one tensor whose steady-state loops have the given relevancies."""
    tensor = _FakeTensor()
    variables = [SimpleNamespace(size=size, relevant=rel, reuse=Reuse.NOT_SET) for size, rel in relevant_sizes]
    space = object.__new__(DecisionSpace)
    space.workload = SimpleNamespace(tensors=[tensor])
    space.ssis = {tensor: SimpleNamespace(get_applicable_temporal_variables=lambda: variables)}
    space.tensors_to_optimize_reuse_for = []
    space.reuse_levels, space.tiles_needed_levels, space.bds_needed_levels = {}, {}, {}
    space.rotation_levels = {}
    space._init_transfer_fire_helpers()
    return tensor, space


def test_double_buffering_skips_loop_invariant_tensors():
    """A loop-invariant tensor (same tile every iteration) reserves one tile, not a double buffer."""
    tensor, space = _fire_helper_stub(relevant_sizes=[(8, False)])
    assert space.tiles_needed_levels[(tensor, -1)] == 1


def test_double_buffering_applies_to_streamed_tensors():
    """An activation tile changes every iteration, so it does need somewhere to prefetch into."""
    tensor, space = _fire_helper_stub(relevant_sizes=[(8, True)])
    assert space.tiles_needed_levels[(tensor, -1)] == 2
