"""Tests for stream.workload.affine_access.

Equivalence gate: derived relevancy over the xDSL operand maps reproduces the framework's
existing (zigzag) R/PR/IR classification on the bundled fixtures. Property tests: footprint
matches brute-force enumeration and is invariant under consistent dimension renaming.
"""

from __future__ import annotations

import itertools
from functools import cache

import pytest
from hypothesis import given
from hypothesis import strategies as st
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineExpr, AffineMap

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.affine_access import (
    compose_dependency,
    footprint,
    map_dim_positions,
    operand_relevancy,
    relevancy,
)
from stream.workload.steady_state.iteration_space import LoopEffect

CONV_FIXTURE = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
SWIGLU_FIXTURE = "stream/inputs/testing/workload/swiglu_1_16_32.onnx"

# A 4x4 output window (full channel/kernel extents, batch 0) for the 2-conv chain.
_CONV_CONSUMER_TILE = {
    0: range(0, 1),
    1: range(0, 4),
    2: range(0, 4),
    3: range(0, 3),
    4: range(0, 3),
    5: range(0, 16),
    6: range(0, 32),
}


@cache
def _workload(onnx_path: str):
    parser = ONNXModelParser(onnx_path)
    parser.run()
    return parser.workload


def _zigzag_relevant_positions(node) -> dict[str, tuple[set[int], dict[int, int]]]:
    """Oracle: the existing framework's R+PR real-dimension positions per operand.

    Builds the zigzag equation/relations from the same affine node via the production adapter,
    then reads LoopRelevancyInfo -- i.e. compares against the code path Stream actually uses.
    """
    from zigzag.datatypes import LayerOperand
    from zigzag.workload.layer_node import LoopRelevancyInfo

    from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator, ZigZagLayerDimRelation

    est = ZigZagCostEstimator(workload=_workload_of(node), accelerator=None, mapping=None)  # type: ignore[arg-type]
    equation, dim_relations, _, _ = est.create_equation_and_dimension_relations_and_padding_and_pr_sizes(node)
    layer_dim_sizes = est.create_layer_dim_sizes(node)
    pr_loop, pr_loop_list, _ = ZigZagLayerDimRelation.extract_pr_loop_info(dim_relations)
    info = LoopRelevancyInfo.extract_relevancy_info(equation, layer_dim_sizes, pr_loop, pr_loop_list)
    sizes = {int(str(k)[1:]): v for k, v in layer_dim_sizes.data.items()}

    op_names = ["O"] + est.input_operand_names[: len(node.inputs)]
    result: dict[str, tuple[set[int], dict[int, int]]] = {}
    for name in op_names:
        operand = LayerOperand(name)
        dims = set(info.get_r_layer_dims(operand))
        for descendants in info.get_pr_layer_dims(operand).values():
            dims |= set(descendants)
        positions = {int(str(d)[1:]) for d in dims if int(str(d)[1:]) < node.num_dims}
        result[name] = (positions, sizes)
    return result


# The workload each node belongs to, needed by the oracle helper.
_NODE_TO_WORKLOAD: dict = {}


def _workload_of(node):
    return _NODE_TO_WORKLOAD[node.name]


@pytest.mark.parametrize("onnx_path", [CONV_FIXTURE, SWIGLU_FIXTURE])
def test_relevancy_matches_zigzag(onnx_path: str):
    """Derived VARYING dims equal zigzag R+PR for every operand (comparing size>1 dims)."""
    workload = _workload(onnx_path)
    input_operand_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for node in workload.get_computation_nodes():
        _NODE_TO_WORKLOAD[node.name] = workload
        oracle = _zigzag_relevant_positions(node)
        op_names = ["O"] + input_operand_names[: len(node.inputs)]
        tensors = (node.outputs[0], *node.inputs)
        for name, tensor in zip(op_names, tensors, strict=True):
            mine = {p for p, effect in operand_relevancy(node, tensor).items() if effect == LoopEffect.VARYING}
            oracle_positions, sizes = oracle[name]
            mine_big = {p for p in mine if sizes.get(p, 2) > 1}
            oracle_big = {p for p in oracle_positions if sizes.get(p, 2) > 1}
            assert mine_big == oracle_big, f"{node.name}/{name}: mine={mine_big} oracle={oracle_big}"


def test_relevancy_classifies_absent_and_invariant():
    """A position outside the node's dims is ABSENT; a node dim absent from the operand is INVARIANT."""
    node = next(iter(_workload(SWIGLU_FIXTURE).get_computation_nodes()))
    assert relevancy(node, node.outputs[0], node.num_dims + 5) == LoopEffect.ABSENT
    effects = set(operand_relevancy(node, node.inputs[0]).values())
    assert effects <= {LoopEffect.VARYING, LoopEffect.INVARIANT}


# --------------------------------------------------------------------------- #
#  Property-based footprint tests                                             #
# --------------------------------------------------------------------------- #
@st.composite
def _affine_map(draw) -> AffineMap:
    num_dims = draw(st.integers(min_value=1, max_value=3))
    num_results = draw(st.integers(min_value=1, max_value=3))
    results: list[AffineExpr] = []
    for _ in range(num_results):
        expr: AffineExpr = AffineExpr.constant(draw(st.integers(min_value=-2, max_value=2)))
        for dim in range(num_dims):
            coeff = draw(st.integers(min_value=-3, max_value=3))
            if coeff:
                expr = expr + AffineExpr.dimension(dim) * coeff
        results.append(expr)
    return AffineMap(num_dims, 0, tuple(results))


@st.composite
def _map_and_box(draw) -> tuple[AffineMap, dict[int, range]]:
    affine_map = draw(_affine_map())
    box: dict[int, range] = {}
    for dim in range(affine_map.num_dims):
        low = draw(st.integers(min_value=-2, max_value=2))
        size = draw(st.integers(min_value=1, max_value=3))
        box[dim] = range(low, low + size)
    return affine_map, box


def _brute_footprint(affine_map: AffineMap, box: dict[int, range]) -> tuple[range, ...]:
    mins = [None] * len(affine_map.results)
    maxs = [None] * len(affine_map.results)
    axes = [list(box[d]) for d in range(affine_map.num_dims)]
    for point in itertools.product(*axes):
        values = affine_map.eval(list(point), [])
        for i, value in enumerate(values):
            mins[i] = value if mins[i] is None else min(mins[i], value)
            maxs[i] = value if maxs[i] is None else max(maxs[i], value)
    return tuple(range(lo, hi + 1) for lo, hi in zip(mins, maxs, strict=True))


@given(_map_and_box())
def test_footprint_matches_bruteforce(map_and_box: tuple[AffineMap, dict[int, range]]):
    affine_map, box = map_and_box
    assert footprint(affine_map, box) == _brute_footprint(affine_map, box)


@given(_map_and_box(), st.randoms())
def test_footprint_invariant_under_renaming(map_and_box: tuple[AffineMap, dict[int, range]], rng):
    affine_map, box = map_and_box
    perm = list(range(affine_map.num_dims))
    rng.shuffle(perm)
    new_dims = [AffineExpr.dimension(perm[d]) for d in range(affine_map.num_dims)]
    renamed = AffineMap(
        affine_map.num_dims,
        0,
        tuple(result.replace_dims_and_symbols(new_dims, []) for result in affine_map.results),
    )
    renamed_box = {perm[d]: box[d] for d in range(affine_map.num_dims)}
    assert footprint(renamed, renamed_box) == footprint(affine_map, box)
    assert map_dim_positions(renamed) == {perm[p] for p in map_dim_positions(affine_map)}


def test_footprint_conv_stride_dilation_handcomputed():
    """1-D conv index ix = stride*ox + dilation*fx over a hand-checked tile."""
    stride, dilation = 2, 3
    index = AffineExpr.dimension(0) * stride + AffineExpr.dimension(1) * dilation
    affine_map = AffineMap(2, 0, (index,))
    # ox in [0, 4), fx in [0, 3): min = 0, max = 2*3 + 3*2 = 12
    tile = {0: range(0, 4), 1: range(0, 3)}
    assert footprint(affine_map, tile) == (range(0, 13),)


def test_footprint_negative_coefficient_and_offset():
    index = AffineExpr.dimension(0) * -1 + AffineExpr.constant(5)
    affine_map = AffineMap(1, 0, (index,))
    assert footprint(affine_map, {0: range(0, 4)}) == (range(2, 6),)  # 5 - [0..3] = [2..5]


def test_footprint_missing_tile_dimension_raises():
    affine_map = AffineMap(2, 0, (AffineExpr.dimension(1),))
    with pytest.raises(ValueError, match="does not bound dimension"):
        footprint(affine_map, {0: range(0, 2)})


def test_non_box_operator_raises():
    """Documented gap: Mod/FloorDiv/CeilDiv indices have no exact box footprint."""
    mod_index = AffineBinaryOpExpr(AffineBinaryOpKind.Mod, AffineExpr.dimension(0), AffineExpr.constant(2))
    affine_map = AffineMap(1, 0, (mod_index,))
    with pytest.raises(NotImplementedError):
        footprint(affine_map, {0: range(0, 4)})


# --------------------------------------------------------------------------- #
#  Dependency composition                                                     #
# --------------------------------------------------------------------------- #
def test_compose_dependency_conv_chain_halo():
    """A consumer output tile pulls a haloed producer region across the 2-conv chain."""
    nodes = list(_workload(CONV_FIXTURE).get_computation_nodes())
    producer, consumer = nodes[0], nodes[1]
    producer_out = producer.get_mapping(producer.outputs[0])
    consumer_in = consumer.get_mapping(consumer.inputs[0])
    region = compose_dependency(producer_out, consumer_in, _CONV_CONSUMER_TILE)
    assert set(region) == set(map_dim_positions(producer_out))
    # batch=1, channel=16, two spatial dims haloed to 4 + (3-1) = 6
    assert sorted(len(r) for r in region.values()) == [1, 6, 6, 16]


def test_compose_dependency_rejects_composite_producer_output():
    """A non-permutation producer output map is out of scope for the box path."""
    producer_out = AffineMap(2, 0, (AffineExpr.dimension(0) + AffineExpr.dimension(1),))
    consumer_in = AffineMap(1, 0, (AffineExpr.dimension(0),))
    with pytest.raises(NotImplementedError):
        compose_dependency(producer_out, consumer_in, {0: range(0, 2)})


# --------------------------------------------------------------------------- #
#  Optional exact (islpy) path                                                #
# --------------------------------------------------------------------------- #
from stream.workload import affine_exact  # noqa: E402

_needs_islpy = pytest.mark.skipif(not affine_exact.is_available(), reason="islpy not installed")


@_needs_islpy
@given(_map_and_box())
def test_exact_footprint_matches_box(map_and_box: tuple[AffineMap, dict[int, range]]):
    affine_map, box = map_and_box
    assert affine_exact.exact_footprint(affine_map, box) == footprint(affine_map, box)


@_needs_islpy
def test_exact_compose_matches_box_on_permutation_chain():
    """On the conv chain (permutation output), the exact path reproduces the box path."""
    nodes = list(_workload(CONV_FIXTURE).get_computation_nodes())
    producer_out = nodes[0].get_mapping(nodes[0].outputs[0])
    consumer_in = nodes[1].get_mapping(nodes[1].inputs[0])
    consumer_tile = _CONV_CONSUMER_TILE
    assert affine_exact.exact_compose_dependency(producer_out, consumer_in, consumer_tile) == compose_dependency(
        producer_out, consumer_in, consumer_tile
    )


@_needs_islpy
def test_exact_compose_handles_composite_output_the_box_rejects():
    """Composite producer output: box path raises; exact path bounds it with a producer domain."""
    producer_out = AffineMap(2, 0, (AffineExpr.dimension(0) + AffineExpr.dimension(1),))
    consumer_in = AffineMap(1, 0, (AffineExpr.dimension(0),))
    consumer_tile = {0: range(2, 5)}  # shared index in [2, 4]
    with pytest.raises(NotImplementedError):
        compose_dependency(producer_out, consumer_in, consumer_tile)
    region = affine_exact.exact_compose_dependency(
        producer_out, consumer_in, consumer_tile, producer_domain={0: range(0, 5), 1: range(0, 5)}
    )
    assert region == {0: range(0, 5), 1: range(0, 5)}
