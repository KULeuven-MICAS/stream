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


def _accelerator():
    data = open_yaml(_SIMD_ACCELERATOR)
    validator = AcceleratorValidator(data, _SIMD_ACCELERATOR)
    validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


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
