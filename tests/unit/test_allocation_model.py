"""The allocator's own part of the model: the families a selection builds, the objective levels they contribute,
the overlap formulation and the buffering helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stream.opt.allocation.constraint_optimization.families import (
    DEFAULT_FAMILIES,
    LATENCY,
    drop_families,
    load_families,
    overlap,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.opt.solver import ObjectiveLevel, PipeliningModel
from stream.workload.steady_state.iteration_space import Reuse

TOTAL_LATENCY = 100


def _objectives(specs=DEFAULT_FAMILIES, **quantities) -> dict[str, ObjectiveLevel]:
    """The objective levels the families ``specs`` contribute to a model with no tensors or transfers, whose
    quantities are ``quantities`` and the overlap's."""
    q = QuantityRegistry()
    for name, expr in {"iteration": 0, "overlap": 0, "fill": 0, **quantities}.items():
        q.add(name, expr)
    model = MagicMock()
    model.add_var.return_value = MagicMock(_raw=TOTAL_LATENCY)
    model.quicksum.return_value = SimpleNamespace(_raw=0)
    space = SimpleNamespace(iterations=1, transfer_nodes=[], links_in_choice={}, tensors_to_optimize_reuse_for=[])
    tta = SimpleNamespace(
        families=load_families(specs),
        context=SimpleNamespace(model=model, space=space, vars=SimpleNamespace(y={}, z_stop={})),
        quantities=q,
    )
    return TransferAndTensorAllocator._objective_levels(tta)  # type: ignore[arg-type]


def test_a_family_left_out_builds_nothing():
    """Leaving a family out of the selection is what switches its constraints and objective terms off."""
    selection = load_families(drop_families(DEFAULT_FAMILIES, ["memory_capacity", "dma_channels"]))
    assert {"memory_capacity", "dma_channels"}.isdisjoint(name for name, _ in selection.steps)
    assert {"memory_capacity", "dma_channels"}.isdisjoint(family.name for family in selection.families)


def test_without_dma_channels_the_primary_objective_is_the_latency():
    """Latency decides first; offchip traffic breaks its ties, buffering breaks traffic's, and the route length
    breaks buffering's."""
    objectives = _objectives(drop_families(DEFAULT_FAMILIES, ["dma_channels"]))
    assert objectives["latency"].expr == TOTAL_LATENCY
    assert list(objectives) == ["latency", "offchip_traffic", "buffering", "route_hops"]
    assert [o.priority for o in objectives.values()] == [4, 3, 2, 1]


def test_dma_channels_charge_their_peaks_in_the_primary_objective():
    assert _objectives(dma_peak_in=3, dma_peak_out=4)["latency"].expr == TOTAL_LATENCY + 7


def test_offchip_traffic_is_charged_in_the_primary_objective():
    """With the family's weight registered, the bytes join the latency in the primary objective."""
    ctx = SimpleNamespace(
        model=MagicMock(quicksum=MagicMock(return_value=SimpleNamespace(_raw=2048))),
        space=SimpleNamespace(tensors_to_optimize_reuse_for=[]),
        vars=SimpleNamespace(z_stop={}),
    )
    q = QuantityRegistry()
    (family,) = load_families(["offchip_traffic"]).families
    assert [level.name for level in family.objective(ctx, q)] == ["offchip_traffic"]
    q.add("offchip_traffic_weight", 1 / 512)
    traffic, latency = family.objective(ctx, q)
    assert (traffic.expr, latency.name, latency.priority, latency.expr) == (2048, "latency", LATENCY, 4)


@pytest.mark.parametrize(
    ("shared_bandwidth", "bandwidth", "charged"),
    [({}, 512.0, True), ({}, 0.0, False), ({0: object()}, 512.0, False)],
    ids=["offchip_links", "no_offchip_core", "shared_bandwidth_model"],
)
def test_offchip_traffic_weight(shared_bandwidth, bandwidth, charged):
    """No off-chip core charges nothing, and with a shared-bandwidth model the bytes are in the latency already."""
    (family,) = load_families(["offchip_traffic"]).families
    space = SimpleNamespace(shared_bandwidth=shared_bandwidth, iterations=4, offchip_bandwidth=lambda: bandwidth)
    q = QuantityRegistry()
    family.build(SimpleNamespace(space=space), q)
    assert ("offchip_traffic_weight" in q) is charged
    if charged:
        assert q.get("offchip_traffic_weight").expr == 4 / bandwidth


def test_the_objective_needs_the_overlap():
    with pytest.raises(ValueError, match="overlap family"):
        _objectives(drop_families(DEFAULT_FAMILIES, ["overlap", "dma_channels"]))


def test_levels_of_one_name_must_share_a_priority():
    class Rogue:
        name = "rogue"

        def objective(self, ctx, q):
            return [ObjectiveLevel(expr=1, priority=LATENCY + 1, name="latency")]

    tta = SimpleNamespace(
        families=SimpleNamespace(families=[*load_families(["dma_channels"]).families, Rogue()]),
        context=None,
        quantities=QuantityRegistry(),
    )
    tta.quantities.add("dma_peak_in", 1)
    tta.quantities.add("dma_peak_out", 1)
    with pytest.raises(ValueError, match="'latency' has priority 5 in 'rogue'"):
        TransferAndTensorAllocator._objective_levels(tta)  # type: ignore[arg-type]


def _overlap(spec) -> PipeliningModel:
    selection = load_families(["reuse_rates", "slot_latency", spec])
    return next(f for f in selection.families if f.name == "overlap").model


def test_pipelining_defaults_to_occupancy():
    """The modulo-scheduling model is the default; span is the opt-in legacy one."""
    assert _overlap("overlap") is PipeliningModel.OCCUPANCY
    assert _overlap({"overlap": {"model": "span"}}) is PipeliningModel.SPAN


@pytest.mark.parametrize(
    ("selected", "double_buffered", "expected"),
    [
        (PipeliningModel.OCCUPANCY, True, PipeliningModel.OCCUPANCY),
        # Overlapping means prefetching the next tile while this one computes -- with a single
        # buffer there is nowhere to prefetch into, so the credit must not be handed out.
        (PipeliningModel.OCCUPANCY, False, PipeliningModel.SPAN),
        (PipeliningModel.SPAN, True, PipeliningModel.SPAN),
        (PipeliningModel.SPAN, False, PipeliningModel.SPAN),
    ],
)
def test_pipelining_requires_double_buffering(selected, double_buffered, expected):
    assert overlap.effective_pipelining(selected, double_buffered) is expected


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


def _fire_helper_stub(*, relevant_sizes, force_double_buffering=True):
    """A decision space carrying one tensor whose steady-state loops have the given relevancies."""
    tensor = _FakeTensor()
    variables = [SimpleNamespace(size=size, relevant=rel, reuse=Reuse.NOT_SET) for size, rel in relevant_sizes]
    space = object.__new__(DecisionSpace)
    space.workload = SimpleNamespace(tensors=[tensor])
    space.ssis = {tensor: SimpleNamespace(get_applicable_temporal_variables=lambda: variables)}
    space.tensors_to_optimize_reuse_for = []
    space.reuse_levels, space.tiles_needed_levels, space.bds_needed_levels = {}, {}, {}
    space.rotation_levels = {}
    space.force_double_buffering = force_double_buffering
    space._init_transfer_fire_helpers()
    return tensor, space


def test_double_buffering_skips_loop_invariant_tensors():
    """A loop-invariant tensor (same tile every iteration) reserves one tile, not a double buffer."""
    tensor, tta = _fire_helper_stub(relevant_sizes=[(8, False)])
    assert tta.tiles_needed_levels[(tensor, -1)] == 1


def test_double_buffering_applies_to_streamed_tensors():
    """An activation tile changes every iteration, so it does need somewhere to prefetch into."""
    tensor, tta = _fire_helper_stub(relevant_sizes=[(8, True)])
    assert tta.tiles_needed_levels[(tensor, -1)] == 2


def test_double_buffering_off_reserves_one_tile():
    tensor, tta = _fire_helper_stub(relevant_sizes=[(8, True)], force_double_buffering=False)
    assert tta.tiles_needed_levels[(tensor, -1)] == 1
