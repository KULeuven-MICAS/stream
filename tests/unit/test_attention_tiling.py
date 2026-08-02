"""The attention head's tiling must respect the softmax's reduction axis.

A softmax reduces the key axis *nonlinearly*: unlike a matmul contraction (a linear reduction that can
be split across cores as partial sums), the softmax needs every element of the key axis before it emits
any output, so the key axis must stay resident -- never spatially unrolled or fusion-split -- in the
conservative (non-flash) model. These tests pin that: the reduced axis is derived as a REDUCTION, the
spatial-unroll guard rejects it, and the generic mapper never splits it while still parallelising the
attention block over its parallel axes (heads/batch) and linear contractions.
"""

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
    nonlinear_reduction_dims,
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
    """The node keeps its fused-kernel identity view (every axis reads PARALLEL), but the declared
    reduction_axes are surfaced by nonlinear_reduction_dims -- the ground truth the tiling guards read."""
    sm = _softmax(build_attention_block())
    assert nonlinear_reduction_dims(sm) == frozenset(sm.reduction_axes)
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
    """End-to-end tiling: across every fused group of the attention head, the softmax's reduced global
    dimension is protected and never inter-core split -- while the block is still split on other axes."""
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
    """The pipeline expands normalizations before mapping, so every guard must read the same answer on
    the expanded graph as on the fused one. Read the declared reduction_axes only and the guards go
    silently dead exactly where they matter -- the expanded graph is the one that gets mapped."""
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
    """A tensor read by both projections has one footprint per accessor once the query and key axes
    are decoupled. Resolving through whichever accessor came first made the reported on-chip tile a
    function of graph construction order rather than of the mapping."""
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
    """'Tiling after fusion' respects the reduction too: the softmax's reduced axis is in the set
    determine_fusion_splits refuses to block-tile (blocking it is online-softmax / flash)."""
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
    """The fix must not over-block: a matmul's *linear* contraction is still spatially splittable (it is
    not in the protected set), so cross-core partial-sum reduction remains available."""
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
    """A large attention head whose score matrix overflows on-chip auto-fuses along the QUERY axis (a
    parallel axis that flows through the intermediates), keeping the key axis (the softmax reduction)
    resident -- the SOTA non-flash fused-attention shape (stream query blocks, keys stay put)."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=4, seq=1024, d_head=64))
    accelerator = _parse_accelerator()
    gen = GenericMappingGenerator(
        accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    sub = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))[0]
    cns = tuple(sub.get_computation_nodes())
    tiling = gen._build_intra_core_tiling(sub, cns)
    assert len(tiling) == 1
    node_name, pos = tiling[0]["dim"].split(".D")
    fusion_dim = sub.get_dims(next(n for n in cns if n.name == node_name))[int(pos)]
    scores = next(n for n in cns if n.name == "scores")
    query, key = sub.get_dims(scores)[2], sub.get_dims(scores)[3]
    assert fusion_dim == query, "the fusion axis must be the query (parallel), not the key"
    assert fusion_dim != key
    assert tiling[0]["tile"] < sub.get_dimension_size(query), "the query is actually blocked (tile < full)"


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
    """The generic pipeline decomposes the softmax first, so verify the *expanded* attention behaves:
    it still fuses into one region (only data-dependent reads cut), and the auto layer-fusion tiling is
    along the query axis -- the key axis is now an ordinary affine reduction (ReduceMax/ReduceSum) that
    is kept resident, not tiled."""
    workload = expand_normalizations(build_attention_block(AttentionConfig(batch=1, heads=4, seq=1024, d_head=64)))
    assert {"ReduceMax", "Exp", "ReduceSum", "Div"} <= {n.type for n in workload.get_computation_nodes()}
    assert determine_fusion_cut_points(workload) == []

    accelerator = _parse_accelerator()
    gen = GenericMappingGenerator(
        accelerator=accelerator, workload=workload, output_dir=tempfile.mkdtemp(), intra_core_tiling=None
    )
    sub = workload.split_fusion_groups(cut_points=determine_fusion_cut_points(workload))[0]
    cns = tuple(sub.get_computation_nodes())
    tiling = gen._build_intra_core_tiling(sub, cns)
    node_name, pos = tiling[0]["dim"].split(".D")
    fusion_dim = sub.get_dims(next(n for n in cns if n.name == node_name))[int(pos)]
    scores = next(n for n in cns if n.name == "scores")
    query, key = sub.get_dims(scores)[2], sub.get_dims(scores)[3]
    assert fusion_dim == query and fusion_dim != key
    assert tiling[0]["tile"] < sub.get_dimension_size(query)


def test_fusion_tiling_plan_describes_query_streaming_softmax_resident():
    """The fusion_tiling_plan API says attention streams the query axis in
    blocks while the softmax key axis stays resident on-chip, and reports the decomposed softmax."""
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
