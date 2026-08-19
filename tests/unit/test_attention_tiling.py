"""The attention head's tiling must respect the softmax's reduction axis."""

from __future__ import annotations

import tempfile

from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.iterator_type import (
    IteratorType,
    NonlinearReductionUnrollError,
    check_spatial_unroll_legal,
    derive_iterator_types,
)
from stream.workload.models import AttentionConfig, build_attention_block
from stream.workload.normalization import expand_normalizations
from stream.workload.workload import determine_fusion_cut_points

_ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def _softmax(workload):
    return next(n for n in workload.get_computation_nodes() if n.type == "Softmax")


def _parse_accelerator():
    ctx = StageContext.from_kwargs(accelerator=_ACCELERATOR, output_path=tempfile.mkdtemp())
    return MainStage([AcceleratorParserStage, LeafStage], ctx).run()[0].get("accelerator")


def test_softmax_reduced_axis_is_tracked_as_a_nonlinear_reduction():
    """The fused-kernel identity view reads every axis PARALLEL; the reduction must be declared."""
    sm = _softmax(build_attention_block())
    assert sm.reduction_axes  # the softmax really does reduce an axis
    # the fused-kernel node view is unchanged: identity, so no axis reads REDUCTION on the node alone
    assert all(t == IteratorType.PARALLEL for t in derive_iterator_types(sm).values())


def test_spatial_unroll_guard_rejects_the_softmax_reduction_axis():
    """A nonlinear reduction cannot be spatially unrolled; a parallel axis can."""
    sm = _softmax(build_attention_block())
    key_axis = sm.reduction_axes[0]
    try:
        check_spatial_unroll_legal(sm, [key_axis])
        raise AssertionError("expected the softmax reduction axis to be rejected for spatial unroll")
    except NonlinearReductionUnrollError:
        pass
    parallel_axis = next(p for p in range(sm.num_dims) if p not in sm.reduction_axes)
    check_spatial_unroll_legal(sm, [parallel_axis])  # must not raise


def test_generic_mapper_never_splits_the_softmax_reduction_axis():
    """Across every fused group the softmax's reduced dim is protected and never inter-core split."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8))
    accelerator = _parse_accelerator()
    gen = GenericMappingGenerator(
        accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    subs = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))

    split_something = False
    for sub in subs:
        cns = tuple(sub.get_computation_nodes())
        protected = gen._protected_dims(sub, cns)
        unroll = gen._inter_core_unrolling(sub, cns)
        for sm in (n for n in cns if n.type == "Softmax"):
            reduced = {sub.get_dims(sm)[p] for p in sm.reduction_axes}
            assert reduced <= protected, f"softmax reduced axis must be protected, got {protected}"
            assert reduced.isdisjoint(unroll), f"softmax reduced axis was inter-core split: {unroll}"
        split_something = split_something or bool(unroll)
    assert split_something, "the mapper must still parallelise the attention block over its other axes"


def test_expansion_does_not_change_what_the_mapper_protects_or_splits():
    """Every guard reads the same protect/split answer on the expanded graph as on the fused one."""
    raw = build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8))
    accelerator = _parse_accelerator()

    def decisions(workload):
        gen = GenericMappingGenerator(
            accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
        )
        return [
            (
                sorted(str(d) for d in gen._protected_dims(sub, cns)),
                sorted(str(d) for d in gen._inter_core_unrolling(sub, cns)),
            )
            for sub in workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))
            if (cns := tuple(sub.get_computation_nodes()))
        ]

    protected_raw = decisions(raw)
    assert any(dims for dims, _ in protected_raw), "the attention block must protect the softmax key axis"
    assert decisions(expand_normalizations(raw)) == protected_raw


def test_tensor_tiles_do_not_depend_on_node_insertion_order():
    """Tensor tiles are a function of the mapping, not of graph construction / node insertion order."""
    from stream.workload.workload import Workload

    accelerator = _parse_accelerator()
    base = expand_normalizations(build_attention_block(AttentionConfig(batch=1, heads=4, seq=1024, d_head=64)))

    def tiles(nodes):
        gen = GenericMappingGenerator(
            accelerator=accelerator, workload=Workload(nodes), output_dir=tempfile.mkdtemp(), intra_core_tiling=None
        )
        return {t["name"]: (tuple(t["tile"]), t["streamed"]) for g in gen.fusion_tiling_plan() for t in g["tensors"]}

    nodes = list(base.nodes)
    reordered = sorted(nodes, key=lambda n: 0 if getattr(n, "name", "") == "proj_q" else 1)
    assert tiles(nodes) == tiles(reordered)


def test_fusion_split_protects_the_softmax_reduction_axis():
    """Tiling after fusion refuses to block-tile the softmax's reduced axis (that would be flash)."""
    from stream.workload.utils import _nonlinear_reduction_group_dims

    workload = build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8))
    group = next(
        sub
        for sub in workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))
        if any(n.type == "Softmax" for n in sub.get_computation_nodes())
    )
    sm = _softmax(group)
    reduced = {group.get_dims(sm)[p] for p in sm.reduction_axes}
    assert reduced <= _nonlinear_reduction_group_dims(group)


def test_linear_contraction_axis_stays_splittable():
    """A matmul's linear contraction stays splittable (not protected), so partial-sum reduction works."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8))
    accelerator = _parse_accelerator()
    gen = GenericMappingGenerator(
        accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    sub = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))[0]
    scores = next(n for n in sub.get_computation_nodes() if n.name == "scores")
    types = derive_iterator_types(scores)
    protected = gen._protected_dims(sub, tuple(sub.get_computation_nodes()))
    linear_reduction_dims = {sub.get_dims(scores)[p] for p, t in types.items() if t == IteratorType.REDUCTION}
    assert linear_reduction_dims, "scores must have a linear contraction dim"
    assert linear_reduction_dims.isdisjoint(protected), "a linear contraction must remain splittable"


def test_auto_fusion_tiles_the_query_axis_when_intermediates_overflow():
    """An overflowing attention head auto-fuses along the query axis, keeping the softmax key resident."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=4, seq=1024, d_head=64))
    gen = GenericMappingGenerator(
        accelerator=_parse_accelerator(), workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    group = gen.fusion_tiling_plan()[0]
    sub = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))[0]
    scores = next(n for n in sub.get_computation_nodes() if n.name == "scores")
    query, key = sub.get_dims(scores)[2], sub.get_dims(scores)[3]

    assert group["streamed_axis"]["name"] == str(query), "the fusion axis must be the query (parallel), not the key"
    assert group["tile"] < group["streamed_axis"]["size"], "the query is actually blocked (tile < full)"
    resident = {a["name"]: a for a in group["resident_axes"]}
    assert str(key) in resident and resident[str(key)]["softmax"], "the softmax key axis stays resident"
    assert str(query) not in resident


def test_small_attention_keeps_the_whole_region_resident():
    """When the whole score matrix fits on-chip, no fusion tiling is emitted (keep it resident)."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=1, seq=8, d_head=8))
    accelerator = _parse_accelerator()
    gen = GenericMappingGenerator(
        accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    sub = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))[0]
    assert gen._auto_fusion_tiling(sub, tuple(sub.get_computation_nodes())) == []


def test_expanded_attention_fuses_and_tiles_the_query_axis():
    """The expanded attention still fuses into one region and auto-tiles the query axis, key resident."""
    workload = expand_normalizations(build_attention_block(AttentionConfig(batch=1, heads=4, seq=1024, d_head=64)))
    assert {"ReduceMax", "Exp", "ReduceSum", "Div"} <= {n.type for n in workload.get_computation_nodes()}
    assert determine_fusion_cut_points(workload) == []

    gen = GenericMappingGenerator(
        accelerator=_parse_accelerator(), workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    group = gen.fusion_tiling_plan()[0]
    sub = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))[0]
    scores = next(n for n in sub.get_computation_nodes() if n.name == "scores")
    query, key = sub.get_dims(scores)[2], sub.get_dims(scores)[3]

    assert group["streamed_axis"]["name"] == str(query)
    assert group["tile"] < group["streamed_axis"]["size"]
    resident = {a["name"] for a in group["resident_axes"]}
    assert str(key) in resident and str(query) not in resident


def test_fusion_tiling_plan_describes_query_streaming_softmax_resident():
    """fusion_tiling_plan streams the query axis in blocks, softmax key resident, softmax decomposed."""
    workload = expand_normalizations(build_attention_block(AttentionConfig(batch=1, heads=4, seq=1024, d_head=64)))
    gen = GenericMappingGenerator(
        accelerator=_parse_accelerator(), workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    plan = gen.fusion_tiling_plan(cut_points=determine_fusion_cut_points(workload))
    assert len(plan) == 1
    group = plan[0]
    assert group["streamed_axis"] is not None
    assert group["tile"] < group["streamed_axis"]["size"], "the query is actually streamed in blocks"
    softmax_axes = [a for a in group["resident_axes"] if a["softmax"]]
    assert softmax_axes, "the softmax reduction axis must be reported as resident"
    assert softmax_axes[0]["size"] == group["streamed_axis"]["size"], "self-attention key seq == query seq"
    assert any(n["fused_kernel"] for n in group["nodes"]), "the softmax is decomposed into tagged sub-ops"
    assert group["buffer_elements"] > 0


def test_streaming_axis_breaks_size_ties_deterministically(monkeypatch):
    """A size tie between streaming-axis candidates resolves by name, deterministically across runs."""
    accelerator = _parse_accelerator()
    workload = expand_normalizations(build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8)))
    gen = GenericMappingGenerator(
        accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    sub = next(iter(workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))))
    cns = tuple(sub.get_computation_nodes())

    tied = sorted(gen._fusible_parallel_dims(sub, cns), key=str)[:2]
    assert len(tied) == 2, "need two fusible parallel axes to tie"
    monkeypatch.setattr(gen, "_fusible_parallel_dims", lambda *_: set(tied))
    monkeypatch.setattr(gen, "_recurrence_dims", lambda *_: set())
    monkeypatch.setattr(sub, "get_dimension_size", lambda dim: 64)  # force the tie

    chosen = gen._streaming_axis(sub, cns, set(tied))
    assert chosen == max(tied, key=str), "a size tie must resolve by name, not by set order"
