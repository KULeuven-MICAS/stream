"""ZigZag's spatial mapping search: the fullest mapping and the best of the others by a quick temporal search, each
searched in full, and the best kept for the mapping objective."""

from types import SimpleNamespace

import pytest
from xdsl.dialects.builtin import i8
from xdsl.ir.affine import AffineMap

from stream.mapping.mapping import Mapping
from stream.stages.estimation.zigzag_cost_estimator import (
    LATENCY_TOLERANCE,
    MappingObjective,
    SpatialMappingSearch,
    ZigZagCostEstimator,
)
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

TPU_LIKE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def _matmul_on_tpu_like() -> tuple[Workload, ComputationNode]:
    a, b, c = (Tensor.create(name, i8, (64, 64)) for name in "ABC")
    maps = tuple(
        AffineMap.from_callable(fn) for fn in (lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n))
    )
    node = ComputationNode(type="MatMul", name="mm", inputs=(a, b), outputs=(c,), operand_mapping=maps)
    return Workload(
        [InEdge(name="a", outputs=(a,)), InEdge(name="b", outputs=(b,)), node, OutEdge(name="c", inputs=(c,))]
    ), node


def _cme(latency: float, energy: float) -> SimpleNamespace:
    return SimpleNamespace(latency_total2=latency, systolic_drain_cycle=0.0, energy_total=energy)


def test_a_matmul_on_a_weight_stationary_array_unrolls_its_outputs_and_contraction():
    """Unrolling the rows and the contraction fills the 32x32 array as fully, but copies every weight into each row's
    registers; the search finds the columns and the contraction, as the array is built for, and so does the quick
    search alone."""
    workload, node = _matmul_on_tpu_like()
    core = parse_accelerator(TPU_LIKE).get_core(0)
    entries = {
        search: ZigZagCostEstimator(workload, None, Mapping(), spatial_mapping_search=search).estimate(node, core)
        for search in SpatialMappingSearch
    }
    fullest, searched = entries[SpatialMappingSearch.FULLEST], entries[SpatialMappingSearch.SHORTLIST]
    assert str(fullest.cme.layer.spatial_mapping) == "{D1: {D0: 32}, D2: {D1: 32}}"
    assert str(searched.cme.layer.spatial_mapping) == "{D1: {D2: 32}, D2: {D1: 32}}"
    assert str(entries[SpatialMappingSearch.QUICK].cme.layer.spatial_mapping) == "{D1: {D2: 32}, D2: {D1: 32}}"
    assert searched.latency_total < fullest.latency_total
    assert searched.energy_total < 0.5 * fullest.energy_total


@pytest.mark.parametrize(
    "objective, best",
    [(MappingObjective.LATENCY, 0), (MappingObjective.ENERGY, 2), (MappingObjective.EDP, 1)],
)
def test_each_objective_ranks_its_best_mapping_first(objective, best):
    workload, _ = _matmul_on_tpu_like()
    cmes = [_cme(100.0, 9.0), _cme(130.0, 6.0), _cme(300.0, 5.0)]
    ranking = ZigZagCostEstimator(workload, None, Mapping(), mapping_objective=objective).ranking()
    assert min(cmes, key=ranking(cmes)) is cmes[best]


def test_of_the_two_kinds_of_spatial_mapping_one_almost_as_fast_wins_on_energy(monkeypatch):
    workload, node = _matmul_on_tpu_like()
    faster, cheaper = _cme(100.0, 9.0), _cme(100.0 * (1 + LATENCY_TOLERANCE), 5.0)
    flows = iter([faster, cheaper])
    estimator = ZigZagCostEstimator(workload, None, Mapping())
    monkeypatch.setattr(estimator, "_array_fill", lambda node, core, mix: 2 if mix else 1)
    monkeypatch.setattr(
        estimator, "instantiate_zigzag_flow", lambda *args: SimpleNamespace(run=lambda: [(next(flows), None)])
    )
    assert estimator.run_zigzag(node, None) is cheaper


@pytest.mark.parametrize("option", ["mapping_objective", "spatial_mapping_search"])
def test_an_unknown_option_value_is_refused(option):
    workload, _ = _matmul_on_tpu_like()
    with pytest.raises(ValueError, match="'fastest' is not a valid"):
        ZigZagCostEstimator(workload, None, Mapping(), **{option: "fastest"})
