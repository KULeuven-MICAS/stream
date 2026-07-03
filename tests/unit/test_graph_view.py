"""Tests for the unified WorkloadGraphView engine (stream.ir.graph_view).

One smart graph view for any workload: proper nodes+edges, repeated-block collapse, fusable regions,
and derived affine metadata per node — the single serialization the platform renders.
"""

from __future__ import annotations

from stream.inputs.testing.workload.make_scan import make_scan_workload
from stream.ir import WorkloadGraphView
from stream.parser.onnx.model import ONNXModelParser
from stream.workload.models import build_attention_block, build_gqa_block, build_kv_cache_decode_step
from stream.workload.rewrites import RewriteParams, get_rewrite


def _view(wl) -> WorkloadGraphView:
    return WorkloadGraphView.from_workload(wl)


def test_repeated_blocks_collapse_into_one_class():
    """The 4 chunks of a chunked scan are the same computation -> one block class ×4, members listed;
    each member node carries that block_class id (so a consumer can draw one and mark the rest ×N)."""
    scan = make_scan_workload().get_computation_nodes()[0]
    chain = get_rewrite("chunked_scan").apply(scan, RewriteParams(chunk_size=2))
    view = _view(chain)
    assert len(view.block_classes) == 1
    cls = view.block_classes[0]
    assert cls.op == "ScanChunk"
    assert cls.multiplicity == 4
    assert cls.representative in cls.members
    for member in cls.members:
        node = next(n for n in view.nodes if n.name == member)
        assert node.block_class == cls.id


def test_attention_projections_collapse():
    """Q/K/V projections are the same MatMul kernel -> one ×3 class; scores/context stay unique."""
    view = _view(build_attention_block())
    matmul_classes = [c for c in view.block_classes if c.op == "MatMul"]
    assert any(c.multiplicity == 3 for c in matmul_classes)


def test_node_kinds_and_regions():
    view = _view(build_kv_cache_decode_step())
    kinds = {n.name: n.kind for n in view.nodes}
    assert kinds["K_valid"] == "data_movement"
    assert kinds["softmax"] == "normalization"
    assert kinds["scores"] == "compute"
    assert any(n.kind == "input" for n in view.nodes) and any(n.kind == "output" for n in view.nodes)
    # every computation node belongs to exactly one region
    assert view.regions
    assert all(n.region is not None for n in view.nodes if n.kind in {"compute", "normalization", "data_movement"})


def test_barrier_kind_from_onnx():
    parser = ONNXModelParser("stream/inputs/testing/workload/attention_head.onnx")
    parser.run()
    view = _view(parser.workload)
    assert any(n.kind == "barrier" and n.op == "Transpose" for n in view.nodes)
    assert len(view.regions) >= 2  # the layout barrier cuts the graph


def test_reuse_captures_gqa_kv_sharing():
    view = _view(build_gqa_block())
    scores = next(n for n in view.nodes if n.name == "scores")
    k_reuse = next(r for r in scores.reuse if r.operand == "k")
    # K is reused across the rep axis (size = reps = 4) — GQA's KV saving
    assert 4 in [a.size for a in k_reuse.axes]


def test_movement_reports_the_moved_slice():
    view = _view(build_kv_cache_decode_step())
    k = next(n for n in view.nodes if n.name == "K_valid")
    assert k.movement is not None
    assert k.movement.kind == "slice" and not k.movement.conservative
    assert k.movement.moved[0].read == (0, 40) and k.movement.moved[0].full == 128


def test_normalization_decomposition_present():
    view = _view(build_attention_block())
    softmax = next(n for n in view.nodes if n.kind == "normalization")
    assert softmax.decomposition is not None
    assert [s.op for s in softmax.decomposition.nodes] == ["ReduceMax", "Exp", "ReduceSum", "Div"]
    assert [s.reduces for s in softmax.decomposition.nodes] == [True, False, True, False]


def test_normalization_decomposition_carries_dimension_propagation():
    """Each sub-op reports its axis roles, so the internal dataflow is inspectable: the key axis is
    REDUCTION in the two reduce sub-ops (contracted) and PARALLEL in exp/div (broadcast back). The
    entry/exit wire the subgraph to the producer/consumer."""
    view = _view(build_attention_block())
    softmax = next(n for n in view.nodes if n.kind == "normalization")
    dec = softmax.decomposition
    assert dec is not None
    key_pos = softmax.reduction_axes[0].pos
    by_op = {s.op: s for s in dec.nodes}
    # the reduced axis is REDUCTION where contracted, PARALLEL (broadcast) where the statistic is reused
    assert by_op["ReduceMax"].dims[key_pos].iterator_type == "REDUCTION"
    assert by_op["ReduceSum"].dims[key_pos].iterator_type == "REDUCTION"
    assert by_op["Exp"].dims[key_pos].iterator_type == "PARALLEL"
    assert by_op["Div"].dims[key_pos].iterator_type == "PARALLEL"
    # the subgraph attaches to the outside at the input-reading and output-producing sub-ops
    assert set(dec.entry) == {"softmax_max", "softmax_exp"}
    assert dec.exit == ["softmax_div"]


def test_untiled_flag_and_json_roundtrip():
    view = _view(build_attention_block())
    assert view.tiled is False
    restored = WorkloadGraphView.model_validate_json(view.model_dump_json())
    assert len(restored.nodes) == len(view.nodes)
    assert restored.block_classes[0].multiplicity == view.block_classes[0].multiplicity


def test_api_entry_point_parses_and_views():
    from stream.api import workload_graph_view

    view = workload_graph_view("stream/inputs/testing/workload/attention_head.onnx")
    assert view["schema_version"] == "1.0"
    assert view["nodes"] and view["edges"]
    assert any(c["op"] == "MatMul" for c in view["block_classes"])
