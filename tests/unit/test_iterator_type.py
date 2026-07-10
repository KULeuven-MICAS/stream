"""Tests for stream.workload.iterator_type: derived PARALLEL/REDUCTION/SEQUENTIAL classification."""

from __future__ import annotations

import numpy as np
import pytest

from stream.inputs.testing.workload.make_scan import ScanConfig, make_scan_workload, scan_reference
from stream.parser.onnx.model import ONNXModelParser
from stream.workload.iterator_type import (
    IteratorType,
    SequentialUnrollError,
    check_spatial_unroll_legal,
    derive_iterator_types,
    is_state_operand,
    sequential_dims,
)

CONV_FIXTURE = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
SWIGLU_FIXTURE = "stream/inputs/testing/workload/swiglu_1_16_32.onnx"


def _workload(path: str):
    parser = ONNXModelParser(path)
    parser.run()
    return parser.workload


@pytest.mark.parametrize("path", [CONV_FIXTURE, SWIGLU_FIXTURE])
def test_existing_workloads_have_no_sequential_dims(path: str):
    """Default path: conv/gemm nodes are only PARALLEL/REDUCTION (bit-identical classification)."""
    for node in _workload(path).get_computation_nodes():
        assert sequential_dims(node) == frozenset()
        assert set(derive_iterator_types(node).values()) <= {IteratorType.PARALLEL, IteratorType.REDUCTION}


def test_gemm_contraction_is_reduction():
    """Gemm O[m,n]=A[m,k]B[k,n]: m,n index the output (PARALLEL), k does not (REDUCTION)."""
    gemm = next(n for n in _workload(SWIGLU_FIXTURE).get_computation_nodes() if n.type == "Gemm")
    types = derive_iterator_types(gemm)
    reductions = [p for p, t in types.items() if t == IteratorType.REDUCTION]
    assert len(reductions) == 1  # exactly the k contraction dim


def test_scan_dim_is_sequential():
    scan = make_scan_workload(ScanConfig(seq_len=8, hidden=16)).get_computation_nodes()[0]
    assert sequential_dims(scan) == frozenset({0})
    types = derive_iterator_types(scan)
    assert types[0] == IteratorType.SEQUENTIAL
    assert types[1] == IteratorType.PARALLEL


def test_state_operand_detection():
    scan = make_scan_workload().get_computation_nodes()[0]
    x, h_prev = scan.inputs
    assert is_state_operand(scan, h_prev) is True
    assert is_state_operand(scan, x) is False


def test_spatial_unroll_of_sequential_dim_raises():
    scan = make_scan_workload().get_computation_nodes()[0]
    with pytest.raises(SequentialUnrollError, match="SEQUENTIAL"):
        check_spatial_unroll_legal(scan, [0])  # t
    check_spatial_unroll_legal(scan, [1])  # d -- legal, no raise


def test_scan_reference_is_prefix_sum():
    x = np.random.default_rng(0).random((6, 4))
    np.testing.assert_allclose(scan_reference(x), np.cumsum(x, axis=0))
