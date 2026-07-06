"""Tests for stream.workload.access_relation (plan/11).

Equivalence gate (permission to proceed): the default :class:`AffineAccess` reproduces
``affine_access.{relevancy, footprint, map_dim_positions}`` for every operand on the swiglu + scan
fixtures, so wrapping the maps in the sum type is bit-identical. Plus piecewise-hull and
data-dependent-parameter behavior.
"""

from __future__ import annotations

from functools import cache

import pytest
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.inputs.testing.workload.make_scan import ScanConfig, make_scan_workload
from stream.parser.onnx.model import ONNXModelParser
from stream.workload.access_relation import (
    AffineAccess,
    DataDependentAccess,
    PiecewiseAffineAccess,
    access_for,
)
from stream.workload.affine_access import footprint, map_dim_positions, relevancy
from stream.workload.steady_state.iteration_space import LoopEffect

SWIGLU_FIXTURE = "stream/inputs/testing/workload/swiglu_1_16_32.onnx"


@cache
def _swiglu_workload():
    parser = ONNXModelParser(SWIGLU_FIXTURE)
    parser.run()
    return parser.workload


def _fixture_nodes():
    """Every ComputationNode of the swiglu (ONNX, affine A/B ops) and scan (SEQUENTIAL) fixtures."""
    return (*_swiglu_workload().get_computation_nodes(), *make_scan_workload(ScanConfig()).get_computation_nodes())


# --------------------------------------------------------------------------- #
#  Equivalence gate: AffineAccess == the derived-access API                   #
# --------------------------------------------------------------------------- #
def test_access_for_is_affine_on_every_fixture_operand():
    """The default derivation is affine for every current node -- bit-identical default path."""
    for node in _fixture_nodes():
        for operand in node.tensors:
            assert isinstance(access_for(node, operand), AffineAccess)


def test_affine_access_indexed_dims_and_relevancy_match_derived_api():
    for node in _fixture_nodes():
        for operand in node.tensors:
            acc = access_for(node, operand)
            assert acc.indexed_dims() == map_dim_positions(node.get_mapping(operand))
            for dim in range(node.num_dims):
                assert acc.relevancy(dim, node.num_dims) == relevancy(node, operand, dim)
            assert acc.relevancy(node.num_dims + 3, node.num_dims) == LoopEffect.ABSENT
            assert acc.is_static


def test_affine_access_footprint_matches_derived_api():
    for node in _fixture_nodes():
        tile = {dim: range(0, 2) for dim in range(node.num_dims)}
        for operand in node.tensors:
            acc = access_for(node, operand)
            assert acc.footprint(tile) == footprint(node.get_mapping(operand), tile)


# --------------------------------------------------------------------------- #
#  Piecewise-affine access                                                    #
# --------------------------------------------------------------------------- #
def test_single_piece_equals_the_wrapped_affine():
    affine = AffineAccess(AffineMap(2, 0, (AffineExpr.dimension(0), AffineExpr.dimension(1))))
    piecewise = PiecewiseAffineAccess((affine,))
    tile = {0: range(0, 3), 1: range(0, 4)}
    assert piecewise.footprint(tile) == affine.footprint(tile)
    assert piecewise.indexed_dims() == affine.indexed_dims()
    assert piecewise.is_static


def test_hull_covers_both_pieces():
    """Piece A reads [0,2]; piece B reads d0+5 -> [5,7]; the hull is [0,7]."""
    piece_a = AffineAccess(AffineMap(1, 0, (AffineExpr.dimension(0),)))
    piece_b = AffineAccess(AffineMap(1, 0, (AffineExpr.dimension(0) + AffineExpr.constant(5),)))
    piecewise = PiecewiseAffineAccess((piece_a, piece_b))
    assert piecewise.footprint({0: range(0, 3)}) == (range(0, 8),)
    assert piecewise.indexed_dims() == {0}


def test_piecewise_rejects_mismatched_rank():
    piece_a = AffineAccess(AffineMap(1, 0, (AffineExpr.dimension(0),)))
    piece_b = AffineAccess(AffineMap(1, 0, (AffineExpr.dimension(0), AffineExpr.constant(0))))
    with pytest.raises(ValueError, match="same operand rank"):
        PiecewiseAffineAccess((piece_a, piece_b))


def test_piecewise_requires_at_least_one_piece():
    with pytest.raises(ValueError, match="at least one"):
        PiecewiseAffineAccess(())


# --------------------------------------------------------------------------- #
#  Data-dependent access                                                      #
# --------------------------------------------------------------------------- #
def test_data_dependent_falls_back_to_bounding_and_is_dynamic():
    bounding = AffineAccess(AffineMap(1, 0, (AffineExpr.dimension(0),)))
    access = DataDependentAccess(bounding=bounding, index_tensor="idx", reuse=0.25)
    tile = {0: range(0, 4)}
    assert access.footprint(tile) == bounding.footprint(tile)
    assert access.indexed_dims() == bounding.indexed_dims()
    assert access.is_static is False
    assert access.reuse == pytest.approx(0.25)
    assert access.relevancy(0, 1) == LoopEffect.VARYING


def test_data_dependent_is_worst_case_when_uncalibrated():
    bounding = AffineAccess(AffineMap(1, 0, (AffineExpr.dimension(0),)))
    access = DataDependentAccess(bounding=bounding, index_tensor="idx")
    assert access.reuse is None
    assert access.footprint({0: range(0, 4)}) == (range(0, 4),)
