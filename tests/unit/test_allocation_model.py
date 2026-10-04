"""The allocator's own part of the model: the families a selection builds, the latency objective, the overlap
formulation and the buffering helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stream.opt.allocation.constraint_optimization.families import DEFAULT_FAMILIES, drop_families, load_families
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.opt.solver import PipeliningModel
from stream.workload.steady_state.iteration_space import Reuse

TOTAL_LATENCY = 100


def _objective_stub(**quantities):
    """A mock allocator whose real objective builder reads ``quantities``."""
    tta = MagicMock(spec=TransferAndTensorAllocator)
    tta.quantities = QuantityRegistry()
    for name, expr in quantities.items():
        tta.quantities.add(name, expr)
    tta.model = MagicMock()
    tta.model.add_var.return_value = MagicMock(_raw=TOTAL_LATENCY)
    tta.model.quicksum.return_value = MagicMock(_raw=0)
    tta.overlap = MagicMock()
    tta.fill = 0
    tta.iterations = 1
    tta.slot_latency = {}
    tta.tensors_to_optimize_reuse_for = []
    tta.transfer_nodes = []
    tta.possible_transfer_allocations = {}
    return tta


def _objectives(tta) -> list:
    TransferAndTensorAllocator._set_total_latency_and_objective(tta)
    return sorted(tta.model.set_lexicographic_objectives.call_args[0][0], key=lambda o: -o.priority)


def test_a_family_left_out_builds_nothing():
    """Leaving a family out of the selection is what switches its constraints off."""
    selection = load_families(drop_families(DEFAULT_FAMILIES, ["memory_capacity", "dma_channels"]))
    assert {"memory_capacity", "dma_channels"}.isdisjoint(name for name, _ in selection.steps)
    tta = MagicMock(spec=TransferAndTensorAllocator)
    tta.object_fifo_depth, tta.bd_depth, tta.shared_bandwidth = {}, {}, {}
    tta._offchip_bandwidth.return_value = 0.0
    for _, build in selection.steps:
        build(tta, QuantityRegistry())
    tta._memory_capacity_constraints.assert_not_called()
    tta._add_dma_usage_constraints.assert_not_called()
    tta._tensor_placement_constraints.assert_called_once()
    tta._overlap.assert_called_once_with(PipeliningModel.OCCUPANCY, True, True)


def test_without_dma_channels_the_primary_objective_is_the_latency():
    """Latency decides first; offchip traffic breaks its ties, and buffering breaks traffic's."""
    objectives = _objectives(_objective_stub())
    assert objectives[0].expr == TOTAL_LATENCY
    assert [o.name for o in objectives] == ["latency", "offchip_traffic", "buffering", "route_hops"]


def test_dma_channels_charge_their_peaks_in_the_primary_objective():
    assert _objectives(_objective_stub(dma_peak_in=3, dma_peak_out=4))[0].expr == TOTAL_LATENCY + 7


def test_offchip_traffic_is_charged_in_the_primary_objective():
    """With the family's weight registered, the bytes join the latency in the primary objective."""
    assert _objectives(_objective_stub(offchip_traffic_weight=1 / 512))[0].expr != TOTAL_LATENCY


@pytest.mark.parametrize(
    ("shared_bandwidth", "bandwidth", "charged"),
    [({}, 512.0, True), ({}, 0.0, False), ({0: object()}, 512.0, False)],
    ids=["offchip_links", "no_offchip_core", "shared_bandwidth_model"],
)
def test_offchip_traffic_weight(shared_bandwidth, bandwidth, charged):
    """No off-chip core charges nothing, and with a shared-bandwidth model the bytes are in the latency already."""
    (family,) = load_families(["offchip_traffic"]).families
    alloc = SimpleNamespace(shared_bandwidth=shared_bandwidth, iterations=4, _offchip_bandwidth=lambda: bandwidth)
    q = QuantityRegistry()
    family.build(alloc, q)
    assert ("offchip_traffic_weight" in q) is charged
    if charged:
        assert q.get("offchip_traffic_weight").expr == 4 / bandwidth


def test_the_objective_needs_the_overlap():
    tta = _objective_stub()
    tta.overlap = None
    with pytest.raises(ValueError, match="overlap family"):
        TransferAndTensorAllocator._set_total_latency_and_objective(tta)


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
    tta = MagicMock(spec=TransferAndTensorAllocator)
    tta.force_double_buffering = double_buffered
    assert TransferAndTensorAllocator._effective_pipelining(tta, selected) is expected


@pytest.mark.parametrize(
    ("model", "expected_builder", "other_builder"),
    [
        (PipeliningModel.OCCUPANCY, "_add_occupancy_indicators", "_add_span_indicators"),
        (PipeliningModel.SPAN, "_add_span_indicators", "_add_occupancy_indicators"),
    ],
)
def test_idle_indicator_dispatch(model, expected_builder, other_builder):
    """Each model builds its own indicators and only its own."""
    tta = MagicMock(spec=TransferAndTensorAllocator)
    tta._resource_activity.return_value = [("res", {0: 0}, "used")]
    TransferAndTensorAllocator._init_idle_indicators(tta, 0, 10, model)
    getattr(tta, expected_builder).assert_called_once()
    getattr(tta, other_builder).assert_not_called()


class _FakeTensor:
    """Hashable stand-in -- the helper keys its dicts by tensor."""

    name = "t"


def _fire_helper_stub(*, relevant_sizes, force_double_buffering=True):
    """A TTA stub carrying one tensor whose steady-state loops have the given relevancies."""
    tensor = _FakeTensor()
    variables = [SimpleNamespace(size=size, relevant=rel, reuse=Reuse.NOT_SET) for size, rel in relevant_sizes]
    tta = MagicMock(spec=TransferAndTensorAllocator)
    tta.workload = SimpleNamespace(tensors=[tensor])
    tta.ssis = {tensor: SimpleNamespace(get_applicable_temporal_variables=lambda: variables)}
    tta.tensors_to_optimize_reuse_for = []
    tta.reuse_levels, tta.tiles_needed_levels, tta.bds_needed_levels = {}, {}, {}
    tta.rotation_levels = {}
    tta.force_double_buffering = force_double_buffering
    TransferAndTensorAllocator._init_transfer_fire_helpers(tta)
    return tensor, tta


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
