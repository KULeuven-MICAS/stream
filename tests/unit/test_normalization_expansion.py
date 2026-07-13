"""Parse-stage decomposition of the softmax into affine sub-operators (M12).

A safe softmax over the key axis is ``max → exp(·−max) → sum → div``: two reduction passes plus two
element-wise broadcasts. As one identity-mapped ``NormalizationNode`` those passes are invisible (the
cost model sees one element-wise op and the reduction axis reads PARALLEL). ``expand_normalizations``
rewrites it, in-graph, into explicit sub-ops so the cost is right and the fusion analysis reads an
ordinary affine reduction; the sub-ops stay tagged so codegen can re-collapse them (round-trip proven).
These tests use the real MHA / GQA attention blocks -- the modern usage of softmax.
"""

from __future__ import annotations

import numpy as np
import pytest

from stream.stages.context import StageContext
from stream.stages.generation.normalization_expansion import ExpandNormalizationStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.models import (
    AttentionConfig,
    GQAConfig,
    build_attention_block,
    build_gqa_block,
)
from stream.workload.node import NormalizationNode
from stream.workload.normalization import (
    REDUCTION_SUBOPS,
    collapse_fused_kernels,
    expand_normalizations,
    softmax_reference,
)
from stream.workload.workload import determine_fusion_cut_points


def _softmax(workload):
    return next(n for n in workload.get_computation_nodes() if isinstance(n, NormalizationNode))


def _by_type(workload):
    return {n.type: n for n in workload.get_computation_nodes()}


def test_expand_softmax_yields_the_safe_softmax_subops():
    """MHA softmax → ReduceMax → Exp → ReduceSum → Div, all tagged with their origin kernel."""
    expanded = expand_normalizations(build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8)))
    by_type = _by_type(expanded)
    assert {"ReduceMax", "Exp", "ReduceSum", "Div"} <= set(by_type)
    for op in ("ReduceMax", "Exp", "ReduceSum", "Div"):
        assert by_type[op].fused_kernel == "Softmax:softmax"
    assert not any(isinstance(n, NormalizationNode) for n in expanded.get_computation_nodes())


def test_reduction_is_explicit_after_expansion():
    """The two reduce sub-ops reduce the key axis as an ordinary affine reduction -- so no softmax
    special-case is needed in the fusion/tiling analysis (contrast: the monolithic node's identity map
    reads the key axis as PARALLEL)."""
    workload = build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8))
    key_axis = _softmax(workload).reduction_axes[0]
    # monolithic node: identity map hides the reduction
    assert derive_iterator_types(_softmax(workload))[key_axis] == IteratorType.PARALLEL
    # expanded: the reduce sub-ops make it a genuine REDUCTION
    by_type = _by_type(expand_normalizations(workload))
    for op in ("ReduceMax", "ReduceSum"):
        types = derive_iterator_types(by_type[op])
        assert [p for p, t in types.items() if t == IteratorType.REDUCTION] == [key_axis]


def test_expansion_counts_two_reduction_passes():
    """The safe softmax has TWO reduction passes (max, sum) -- the fidelity a single identity-mapped
    node under-counts to one element-wise pass."""
    expanded = expand_normalizations(build_attention_block())
    reductions = [n for n in expanded.get_computation_nodes() if n.type in REDUCTION_SUBOPS]
    assert sorted(n.type for n in reductions) == ["ReduceMax", "ReduceSum"]


def test_expansion_reconnects_downstream_consumers():
    """The final Div writes the softmax's original output tensor, so the context matmul is unchanged."""
    expanded = expand_normalizations(build_attention_block())
    div = _by_type(expanded)["Div"]
    context = next(n for n in expanded.get_computation_nodes() if n.name == "context")
    assert div.outputs[0].name in {t.name for t in context.inputs}


def test_expanded_attention_still_fuses_into_one_region():
    """Expansion is a cost/fusion refinement, not a barrier: the block still has no data-dependent read,
    so it fuses into one region (now with the reduction passes explicit)."""
    expanded = expand_normalizations(build_attention_block())
    assert determine_fusion_cut_points(expanded) == []


@pytest.mark.parametrize(
    "workload",
    [
        build_attention_block(AttentionConfig(batch=1, heads=2, seq=8, d_head=8)),  # 4-D scores
        build_gqa_block(GQAConfig(groups=2, reps=2, seq=8, d_head=8)),  # 5-D scores (grouped heads)
    ],
    ids=["mha", "gqa"],
)
def test_expand_collapse_round_trip(workload):
    """collapse(expand(·)) reconstructs the exact softmax -- the tagged sub-op representation is
    sufficient to rebuild the native kernel for codegen (type, name, reduction axis, tensors)."""
    original = _softmax(workload)
    collapsed = collapse_fused_kernels(expand_normalizations(workload))
    restored = _softmax(collapsed)
    assert restored.type == original.type
    assert restored.name == original.name
    assert restored.reduction_axes == original.reduction_axes
    assert restored.inputs[0].name == original.inputs[0].name
    assert restored.outputs[0].name == original.outputs[0].name
    assert restored.num_dims == original.num_dims


def test_stage_is_opt_in():
    """ExpandNormalizationStage only expands when the flag is set (default: monolithic softmax)."""
    workload = build_attention_block()

    def run(expand: bool):
        ctx = StageContext.from_kwargs(workload=workload, expand_normalizations=expand)
        return list(MainStage([ExpandNormalizationStage, LeafStage], ctx).run())[0].get("workload")

    assert any(isinstance(n, NormalizationNode) for n in run(False).get_computation_nodes())
    assert not any(isinstance(n, NormalizationNode) for n in run(True).get_computation_nodes())


def test_subops_mirror_the_numpy_softmax_reference():
    """The sub-op math is the safe softmax the reference computes (guards the decomposition semantics)."""
    x = np.random.default_rng(0).standard_normal((2, 4, 8, 8)).astype(np.float32)
    ref = softmax_reference(x, axis=3)
    manual_max = x - x.max(axis=3, keepdims=True)
    manual = np.exp(manual_max) / np.exp(manual_max).sum(axis=3, keepdims=True)
    assert np.allclose(ref, manual, atol=1e-6)
    assert np.allclose(ref.sum(axis=3), 1.0, atol=1e-6)
