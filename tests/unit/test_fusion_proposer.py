"""Auto-checks for the fusion-region proposer.

The mandated checks: every proposed region is legal by the affine analysis (no data-dependent edge
sits inside a region), buffer predictions match hand-derived values on conv/attention/scan fixtures,
known-good regions are recovered, and capacity/recurrence behavior is correct.
"""

from __future__ import annotations

from functools import cache

import pytest

from stream.parser.onnx.model import ONNXModelParser
from stream.stages.context import StageContext
from stream.stages.generation.fusion_proposal import FusionProposalStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.blocks import build_block
from stream.workload.fusion.analysis import workload_fusion_edges
from stream.workload.fusion.proposer import edge_stream_buffer, propose_fusion_regions

CONV_FIXTURE = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
UNBOUNDED = 2**60


@cache
def _conv_workload():
    parser = ONNXModelParser(CONV_FIXTURE)
    parser.run()
    return parser.workload


def _node(workload, name):
    return next(n for n in workload.get_computation_nodes() if n.name == name)


def _shared(producer, consumer):
    return next(t for t in producer.outputs if t in consumer.inputs)


def _region_of(regions, node_name):
    return next(i for i, r in enumerate(regions) if node_name in r.nodes)


def _is_legal(workload, regions) -> bool:
    """No data-dependent edge may have both endpoints in the same region."""
    member = {name: i for i, r in enumerate(regions) for name in r.nodes}
    return not any(
        e.data_dependent and member.get(e.producer) == member.get(e.consumer) for e in workload_fusion_edges(workload)
    )


# --------------------------------------------------------------------------- #
#  Legality: every region is legal by the affine analysis (all blocks)        #
# --------------------------------------------------------------------------- #
BLOCK_KEYS = ["attention", "gqa", "kv_cache", "swiglu", "rmsnorm", "moe", "chunked_ssm", "flash_attention"]


@pytest.mark.parametrize("key", BLOCK_KEYS)
def test_every_proposed_region_is_legal(key: str):
    wl = build_block(key)
    regions = propose_fusion_regions(wl, UNBOUNDED)
    assert regions
    assert _is_legal(wl, regions)
    assert {n for r in regions for n in r.nodes} == {n.name for n in wl.get_computation_nodes()}  # partition


def test_conv_chain_region_is_legal():
    wl = _conv_workload()
    assert _is_legal(wl, propose_fusion_regions(wl, UNBOUNDED))


# --------------------------------------------------------------------------- #
#  Buffer predictions match hand-derived values                               #
# --------------------------------------------------------------------------- #
def test_scan_state_carry_buffer_is_o1():
    """A chunked recurrence streams with an O(1) state carry (window 1) -- independent of sequence."""
    wl = build_block("chunked_ssm", seq=64, hidden=16, chunk_size=16)  # 4 chunks
    chunk0, chunk1 = _node(wl, "ssm_scan_chunk0"), _node(wl, "ssm_scan_chunk1")
    assert edge_stream_buffer(chunk0, chunk1, _shared(chunk0, chunk1)) == 1


def test_attention_softmax_buffer_is_the_full_key_row():
    """Streaming attention over queries, softmax must hold the full key row (length seq)."""
    wl = build_block("attention", batch=1, heads=1, seq=4, d_head=2)
    scores, softmax = _node(wl, "scores"), _node(wl, "softmax")
    assert edge_stream_buffer(scores, softmax, _shared(scores, softmax)) == 4  # == seq


def test_conv_chain_buffer_is_a_bounded_line_buffer():
    """A conv->conv fusion holds a line buffer + halo, far smaller than the full feature map."""
    wl = _conv_workload()
    convs = wl.get_computation_nodes()
    producer, consumer = convs[0], convs[1]
    shared = _shared(producer, consumer)
    full = 1
    for size in shared.shape:
        full *= size
    buffer = edge_stream_buffer(producer, consumer, shared)
    assert 0 < buffer < full  # a bounded line buffer, not the whole materialized feature map


# --------------------------------------------------------------------------- #
#  Known-good regions recovered                                               #
# --------------------------------------------------------------------------- #
def test_attention_is_one_fused_region():
    wl = build_block("attention")
    regions = propose_fusion_regions(wl, UNBOUNDED)
    assert len(regions) == 1
    assert regions[0].boundary_reason == "sink"


def test_swiglu_is_one_fused_region():
    assert len(propose_fusion_regions(build_block("swiglu"), UNBOUNDED)) == 1


def test_conv_chain_is_one_fused_region():
    assert len(propose_fusion_regions(_conv_workload(), UNBOUNDED)) == 1


def test_moe_splits_at_the_data_dependent_combine():
    wl = build_block("moe", tokens=8, experts=2, capacity=4)
    regions = propose_fusion_regions(wl, UNBOUNDED)
    # the expert MLP fuses; the data-dependent combine is a separate region (legal split)
    assert _region_of(regions, "expert_out") != _region_of(regions, "combine")
    assert _region_of(regions, "dispatch") == _region_of(regions, "expert_out")
    combine_region = regions[_region_of(regions, "combine")]
    assert combine_region.boundary_reason in ("data_dependent", "sink")
    assert _is_legal(wl, regions)


# --------------------------------------------------------------------------- #
#  Capacity + recurrence prioritization                                       #
# --------------------------------------------------------------------------- #
def test_capacity_splits_a_region_monotonically():
    wl = build_block("attention", batch=1, heads=1, seq=8, d_head=4)
    unbounded = len(propose_fusion_regions(wl, UNBOUNDED))
    tight = len(propose_fusion_regions(wl, 1))  # below the softmax key-row buffer -> must split
    assert unbounded == 1
    assert tight > unbounded


def test_recurrence_chain_stays_fused_under_tiny_capacity():
    """Recurrence is prioritized: an O(1)-state chain fuses even at capacity 1, where a materialized
    tensor would not."""
    wl = build_block("chunked_ssm", seq=64, hidden=16, chunk_size=16)  # 4 chunks, state carry buffer = 1
    assert len(propose_fusion_regions(wl, 1)) == 1  # whole chain stays fused
    assert len(propose_fusion_regions(wl, 0)) == 4  # capacity 0 forbids even the O(1) carry


def test_fusion_proposal_stage_annotates_context():
    ctx = StageContext.from_kwargs(workload=build_block("attention"))
    result = MainStage([FusionProposalStage, LeafStage], ctx).run()
    assert len(result) == 1
    regions = result[0].get("proposed_fusion_regions")
    assert regions and len(regions) == 1
