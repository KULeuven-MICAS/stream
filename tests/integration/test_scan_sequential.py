"""Integration tests for the sequential-scan path: the cost model runs on the scan, and the
tiling pipeline refuses to spatially unroll the SEQUENTIAL dimension."""

from __future__ import annotations

import pytest
from zigzag.utils import open_yaml

from stream.inputs.testing.workload.make_scan import ScanConfig, make_scan_workload
from stream.mapping.mapping import Mapping, NodeMapping
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.workload.iterator_type import SequentialUnrollError
from stream.workload.utils import collect_spatial_unrollings

_SIMD_ACCELERATOR = "stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml"


def _accelerator_from(path: str):
    data = open_yaml(path)
    validator = AcceleratorValidator(data, path)
    validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


def _accelerator():
    return _accelerator_from(_SIMD_ACCELERATOR)


def test_scan_costs_end_to_end_on_simd_arch():
    """The scan node (state read at t-1) costs through the real ZigZag cost model on a SIMD core."""
    workload = make_scan_workload(ScanConfig(seq_len=8, hidden=16))
    scan = workload.get_computation_nodes()[0]
    accelerator = _accelerator()
    estimator = ZigZagCostEstimator(
        workload=workload, accelerator=accelerator, mapping=Mapping(initial={scan: NodeMapping()})
    )
    compute_core = next(c for c in accelerator.core_list if "compute" in str(c.core_type))
    entry = estimator.estimate(scan, compute_core)
    assert entry.latency_total > 0
    assert entry.cme is not None, "expected a real cost-model evaluation, not the ideal-cycle fallback"


def test_tiling_rejects_spatial_unroll_of_sequential_dim():
    """A mapping that inter-core splits the sequential dim is rejected; splitting the parallel dim is fine."""
    workload = make_scan_workload(ScanConfig(seq_len=8, hidden=16))
    scan = workload.get_computation_nodes()[0]
    seq_dim, parallel_dim = workload.get_dims(scan)  # (t, d)

    bad = Mapping(initial={scan: NodeMapping(inter_core_tiling=(((seq_dim, 2),),))})
    with pytest.raises(SequentialUnrollError, match="SEQUENTIAL"):
        collect_spatial_unrollings(workload, bad)

    ok = Mapping(initial={scan: NodeMapping(inter_core_tiling=(((parallel_dim, 2),),))})
    collect_spatial_unrollings(workload, ok)  # no raise


def test_the_ideal_cycle_fallback_survives_a_core_with_no_modelled_array():
    """An aie2 tile describes no `operational_array` at all, and `Core.__getattr__` raises rather
    than returning None for it.

    The fallback exists precisely for cores ZigZag cannot map, so reaching for an attribute that
    kind of core does not have turned "this array is not modelled" into an AttributeError that took
    the whole run down — on the one hardware the fallback was written to serve.
    """
    from stream.hardware.architecture.core import Core  # noqa: PLC0415

    accelerator = _accelerator_from("stream/inputs/aie/hardware/single_core.yaml")
    tile = next(c for c in accelerator.core_list if "compute" in str(c.core_type))
    assert isinstance(tile, Core)
    with pytest.raises(AttributeError):
        _ = tile.operational_array  # the precondition this test exists for

    workload = make_scan_workload(ScanConfig(seq_len=8, hidden=16))
    scan = workload.get_computation_nodes()[0]
    estimator = ZigZagCostEstimator(
        workload=workload, accelerator=accelerator, mapping=Mapping(initial={scan: NodeMapping()})
    )
    entry = estimator.estimate(scan, tile)
    assert entry.latency_total > 0
    # `cme is None` is the honest report: nothing about this core's memory hierarchy was modelled.
    assert entry.cme is None
