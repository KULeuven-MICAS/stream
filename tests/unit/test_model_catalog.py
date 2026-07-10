"""Tests for the canonical architecture builders (stream.workload.models).

These pin the *internal representation* the framework showcases: attention is affine MatMuls with
a single Softmax barrier; Mamba is a SEQUENTIAL recurrence that chunks. They are representation
tests, not numerical ones (the chunked-scan math is verified in tests/rewrites).
"""

from __future__ import annotations

from stream.workload.iterator_type import IteratorType, derive_iterator_types, sequential_dims
from stream.workload.models import MODEL_CATALOG, build_attention_block, build_mamba_block
from stream.workload.node import FusionEdge, NormalizationNode
from stream.workload.normalization import parallel_axes, reduction_axes
from stream.workload.rewrites import RewriteParams, get_rewrite


def test_catalog_builds_every_model():
    keys = {s.key for s in MODEL_CATALOG}
    assert {"attention", "gqa", "linear_attention", "mamba", "kv_cache"} <= keys
    for spec in MODEL_CATALOG:
        wl = spec.build()
        assert wl.get_computation_nodes(), f"{spec.key} produced no compute nodes"


def test_attention_softmax_is_a_fusible_normalization_not_a_barrier():
    wl = build_attention_block()
    assert [n for n in wl.nodes if isinstance(n, FusionEdge)] == []  # no barriers
    softmax = next(n for n in wl.get_computation_nodes() if isinstance(n, NormalizationNode))
    # reduces the key axis (position 3), parallel over batch/head/query
    assert reduction_axes(softmax) == (3,)
    assert parallel_axes(softmax) == (0, 1, 2)
    # the affine MatMuls each contract exactly the expected axes; none is recurrent
    for n in wl.get_computation_nodes():
        if isinstance(n, NormalizationNode):
            continue
        assert n.type == "MatMul"
        assert any(t == IteratorType.REDUCTION for t in derive_iterator_types(n).values())
        assert sequential_dims(n) == frozenset()


def test_attention_is_one_fusible_region():
    """Softmax fuses (parallel over b,h,i), so the whole block is a single fusible region containing
    the projections, scores, softmax and context/output — the flash-attention view."""
    wl = build_attention_block()
    groups = wl.split_fusion_groups()
    assert len(groups) == 1
    names = {c.name for c in groups[0].get_computation_nodes()}
    assert {"scores", "softmax", "context"} <= names
    assert len(groups[0].get_dimension_sizes()) > 0


def test_mamba_scan_is_sequential_and_chunks():
    wl = build_mamba_block()
    scan = next(n for n in wl.get_computation_nodes() if n.type == "Scan")
    assert sequential_dims(scan)  # the sequence axis carries state
    assert IteratorType.SEQUENTIAL in derive_iterator_types(scan).values()
    # the chunked rewrite decomposes the scan into a per-chunk reduction chain
    rewrite = get_rewrite("chunked_scan")
    assert rewrite.matches(scan)
    chain = rewrite.apply(scan, RewriteParams(chunk_size=32))
    assert len(chain.get_computation_nodes()) >= 1
