"""M04 rewrite library: golden numerics (chunked == direct) and subgraph structure vs chunk size."""

from __future__ import annotations

from math import ceil

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.workload.node import ComputationNode
from stream.workload.rewrites import apply_rewrites, get_rewrite, registered_rewrites
from stream.workload.rewrites.base import RewriteParams
from stream.workload.rewrites.reference import (
    chunked_deltanet,
    chunked_diagonal_scan,
    chunked_ssd,
    direct_deltanet,
    direct_diagonal_scan,
    direct_ssd,
)
from stream.workload.tensor import Tensor

_RNG = np.random.default_rng(20260702)


def _source(type_name: str, seq_len: int, hidden: int) -> ComputationNode:
    x = Tensor.create("x", bf16, (seq_len, hidden))
    y = Tensor.create("y", bf16, (seq_len, hidden))
    identity = AffineMap.from_callable(lambda t, d: (t, d))
    return ComputationNode(
        type=type_name, name=f"{type_name.lower()}_src", inputs=(x,), outputs=(y,), operand_mapping=(identity, identity)
    )


def _compute_chain_edges(workload) -> int:
    return sum(1 for u, v in workload.edges() if isinstance(u, ComputationNode) and isinstance(v, ComputationNode))


# --------------------------------------------------------------------------- #
#  Golden numerics: chunked decomposition == direct recurrence                #
# --------------------------------------------------------------------------- #
@given(
    length=st.integers(min_value=1, max_value=24),
    dim=st.integers(min_value=1, max_value=6),
    chunk_size=st.integers(min_value=1, max_value=24),
)
def test_chunked_scan_matches_direct(length: int, dim: int, chunk_size: int):
    x = _RNG.standard_normal((length, dim))
    a = _RNG.uniform(0.9, 1.0, (length, dim))
    b = _RNG.standard_normal((length, dim))
    np.testing.assert_allclose(chunked_diagonal_scan(x, a, b, chunk_size), direct_diagonal_scan(x, a, b), atol=1e-9)


@pytest.mark.parametrize("length,state,chunk_size", [(12, 4, 3), (16, 6, 5), (9, 3, 9), (20, 5, 1)])
def test_chunked_ssd_matches_direct(length: int, state: int, chunk_size: int):
    x = _RNG.standard_normal(length)
    a = _RNG.uniform(0.9, 1.0, length)
    b_mat = _RNG.standard_normal((length, state))
    c_mat = _RNG.standard_normal((length, state))
    np.testing.assert_allclose(chunked_ssd(x, a, b_mat, c_mat, chunk_size), direct_ssd(x, a, b_mat, c_mat), atol=1e-8)


@pytest.mark.parametrize("length,dk,dv,chunk_size", [(10, 3, 2, 3), (12, 4, 4, 5), (8, 2, 3, 8)])
def test_chunked_deltanet_matches_direct(length: int, dk: int, dv: int, chunk_size: int):
    q = _RNG.standard_normal((length, dk))
    k = _RNG.standard_normal((length, dk))
    v = _RNG.standard_normal((length, dv))
    alpha = _RNG.uniform(0.9, 1.0, length)
    beta = _RNG.uniform(0.0, 1.0, length)
    np.testing.assert_allclose(
        chunked_deltanet(q, k, v, alpha, beta, chunk_size), direct_deltanet(q, k, v, alpha, beta), atol=1e-9
    )


# --------------------------------------------------------------------------- #
#  Structure: subgraph shape and dependency chain vs chunk size               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rewrite_name,source_type", [("chunked_scan", "Scan"), ("ssd", "SSD"), ("gated_deltanet", "GatedDeltaNet")]
)
def test_rewrite_produces_chunk_chain(rewrite_name: str, source_type: str):
    seq_len, hidden, chunk_size = 12, 8, 4
    node = _source(source_type, seq_len, hidden)
    workload = get_rewrite(rewrite_name).apply(node, RewriteParams(chunk_size=chunk_size))

    chunks = sorted(workload.get_computation_nodes(), key=lambda n: n.name)
    n_chunks = ceil(seq_len / chunk_size)
    assert len(chunks) == n_chunks
    # inter-chunk dependency chain has exactly n_chunks - 1 edges
    assert _compute_chain_edges(workload) == n_chunks - 1
    # each chunk node reduces over its chunk-local sequence extent
    for i, chunk in enumerate(chunks):
        assert chunk.inputs[0].shape[0] == min(chunk_size, seq_len - i * chunk_size)


def test_chunk_size_sweep_scales_chain_length():
    node = _source("Scan", seq_len=24, hidden=8)
    counts = []
    for chunk_size in (2, 3, 4, 6, 12, 24):
        workload = get_rewrite("chunked_scan").apply(node, RewriteParams(chunk_size=chunk_size))
        counts.append(len(workload.get_computation_nodes()))
        assert counts[-1] == ceil(24 / chunk_size)
    assert counts == sorted(counts, reverse=True)  # larger chunks -> fewer (monotone) chunk nodes


def test_registry_and_dispatch():
    assert registered_rewrites() == ["chunked_scan", "gated_deltanet", "ssd"]
    scan = _source("Scan", 8, 4)
    assert apply_rewrites(scan, RewriteParams(chunk_size=4)) is not None
    assert apply_rewrites(_source("Conv", 8, 4), RewriteParams(chunk_size=4)) is None


def test_rewrite_params_rejects_nonpositive_chunk():
    with pytest.raises(ValueError, match="chunk_size"):
        RewriteParams(chunk_size=0)
