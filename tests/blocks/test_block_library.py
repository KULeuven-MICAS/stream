"""Tests for the registry-driven block library (plan Phase 2).

Every block builds; the catalog is reused (not duplicated); and the modern blocks carry the right
AccessRelation types -- MoE dispatch/combine are DataDependent, RMSNorm reduces, chunked SSM chunks.
"""

from __future__ import annotations

import pytest

from stream.workload.access_relation import AffineAccess, DataDependentAccess, PiecewiseAffineAccess, access_for
from stream.workload.blocks import available_blocks, build_block, get_block
from stream.workload.blocks.library import sparse_attention_key_access
from stream.workload.node import NormalizationNode

EXPECTED_KEYS = {
    "attention",
    "gqa",
    "linear_attention",
    "mamba",
    "kv_cache",
    "swiglu",
    "rmsnorm",
    "moe",
    "chunked_ssm",
    "flash_attention",
}


def _node(workload, name):
    return next(n for n in workload.get_computation_nodes() if n.name == name)


def test_registry_exposes_catalog_and_modern_blocks():
    keys = {b.key for b in available_blocks()}
    assert EXPECTED_KEYS <= keys


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_every_block_builds_with_compute_nodes(key: str):
    workload = build_block(key)
    assert workload.get_computation_nodes(), f"{key} produced no compute nodes"


def test_block_params_pass_through_and_reject_typos():
    small = build_block("attention", heads=2, seq=16)
    assert small.get_computation_nodes()
    with pytest.raises(TypeError):
        build_block("attention", not_a_real_param=1)


def test_swiglu_has_affine_projections_and_elementwise():
    wl = build_block("swiglu")
    types = {n.name: n.type for n in wl.get_computation_nodes()}
    assert types["gate_proj"] == "MatMul" and types["down_proj"] == "MatMul"
    assert types["silu"] == "Silu" and types["gate_mul"] == "Mul"


def test_rmsnorm_is_a_reduction_normalization():
    wl = build_block("rmsnorm")
    norm = _node(wl, "rmsnorm")
    assert isinstance(norm, NormalizationNode)
    assert norm.reduction_axes == (1,)


def test_moe_dispatch_and_combine_read_data_dependently():
    wl = build_block("moe", tokens=8, experts=2, capacity=4)
    dispatch = _node(wl, "dispatch")
    combine = _node(wl, "combine")
    # the *data* input of dispatch/combine is a DataDependentAccess with the router as its index
    dispatch_access = access_for(dispatch, dispatch.inputs[0])
    combine_access = access_for(combine, combine.inputs[0])
    assert isinstance(dispatch_access, DataDependentAccess)
    assert isinstance(combine_access, DataDependentAccess)
    assert dispatch_access.index_tensor == "router_logits"
    assert dispatch_access.is_static is False
    # the routing logits (a non-data input) and the affine expert GEMM stay affine
    assert isinstance(access_for(dispatch, dispatch.inputs[1]), AffineAccess)
    assert isinstance(access_for(_node(wl, "expert_in"), _node(wl, "expert_in").inputs[0]), AffineAccess)


def test_moe_capacity_is_a_sweep_variable():
    small = build_block("moe", tokens=8, experts=2, capacity=4)
    large = build_block("moe", tokens=8, experts=2, capacity=8)
    dispatched_small = _node(small, "dispatch").outputs[0].shape
    dispatched_large = _node(large, "dispatch").outputs[0].shape
    assert dispatched_small[1] == 4 and dispatched_large[1] == 8


def test_chunked_ssm_chain_length_tracks_chunk_size():
    coarse = build_block("chunked_ssm", seq=64, hidden=16, chunk_size=32)  # ceil(64/32) = 2 chunks
    fine = build_block("chunked_ssm", seq=64, hidden=16, chunk_size=16)  # ceil(64/16) = 4 chunks
    n_coarse = len(coarse.get_computation_nodes())
    n_fine = len(fine.get_computation_nodes())
    assert n_fine > n_coarse
    assert all(n.type == "ScanChunk" for n in fine.get_computation_nodes())


def test_sparse_attention_key_access_is_piecewise_affine():
    """Sparse (local + dilated) attention keys are a union of two affine bands -> PiecewiseAffine."""
    access = sparse_attention_key_access(window=4, dilation=2)
    assert isinstance(access, PiecewiseAffineAccess)
    assert access.is_static  # structured sparsity is static, not data-dependent
    # query i=8, band offset w in [0,4): local band 8-4+[0,3]=[4,7]; dilated 8-2*[0,3]=[2,8]; hull [2,8]
    assert access.footprint({0: range(8, 9), 1: range(0, 4)}) == (range(2, 9),)


def test_flash_attention_block_is_a_dense_per_block_reduction_scan():
    """Flash attention decomposes into AttentionBlock nodes, each a dense reduction over its key block
    (query/head PARALLEL, block-key REDUCTION) -- the online-softmax scan."""
    from stream.workload.iterator_type import IteratorType, derive_iterator_types

    wl = build_block("flash_attention", seq_q=8, seq_k=32, d_head=16, block_size=8)  # 4 key blocks
    blocks = [n for n in wl.get_computation_nodes() if n.type == "AttentionBlock"]
    assert len(blocks) == 4
    roles = set(derive_iterator_types(blocks[0]).values())
    assert IteratorType.REDUCTION in roles and IteratorType.PARALLEL in roles


def test_get_block_unknown_key_raises():
    with pytest.raises(KeyError):
        get_block("does_not_exist")
